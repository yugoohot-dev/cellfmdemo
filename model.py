import torch
from torch import nn
from mindspore.train.serialization import load_checkpoint
from scipy.sparse import csr_matrix
from layers.torch_finetune import FinetuneModel
from layers.utils import *
import torch.nn.functional as F
from layers.torch_model import SpeciesAwareMoE, ortholog_alignment_loss


class Cell_FM(nn.Module):
    def __init__(self, n_gene, cfg, ckpt_path=None, device=None):
        super().__init__()
        self.cfg = cfg
        self.ckpt_path = ckpt_path
        self.ms_ckpt = None
        self.device = device
        self.optimizer = None
        self.scaler = None
        self.moment_state_dict_1 = None
        self.moment_state_dict_2 = None
        self.meta_info = None
        self.net = FinetuneModel(n_gene, self.cfg)
        self.params = (
            list(self.net.parameters())
        )
        # [新增] 实例化 MoE 网络 (基因级 + 细胞级)
        # ==========================================
        self.gene_moe = SpeciesAwareMoE(dim=self.cfg.enc_dims, num_species=6, is_cell_level=False)
        self.cell_moe = SpeciesAwareMoE(dim=self.cfg.enc_dims, num_species=6, is_cell_level=True)
        
        self.params.extend(list(self.gene_moe.parameters()))
        self.params.extend(list(self.cell_moe.parameters()))
        
    def load_model(self, weight, moment):
        if weight:
            self.ms_ckpt = load_checkpoint(self.ckpt_path)
            self.load_weight()
        self.init_optimizer(moment=moment)
        return self, self.optimizer, self.scaler
    
    def map_ms_to_pt(self, ms_key):
        name = ms_key
        name = name.replace("layer_norm.gamma", "weight")
        name = name.replace("layer_norm.beta", "bias")
        name = name.replace("post_norm1.gamma", "post_norm1.weight")
        name = name.replace("post_norm1.beta", "post_norm1.bias")
        name = name.replace("post_norm2.gamma", "post_norm2.weight")
        name = name.replace("post_norm2.beta", "post_norm2.bias")
        return name

    def load_weight(self):
        torch_state_dict = {}
        moment_state_dict_1 = {}
        moment_state_dict_2 = {}
        for ms_key, ms_param in self.ms_ckpt.items():
            pt_key = self.map_ms_to_pt(ms_key)
            pt_tensor = torch.tensor(ms_param.asnumpy())
            if pt_key.startswith("moment1."):
                param_name = pt_key[len("moment1."):]
                moment_state_dict_1[param_name] = pt_tensor
            elif pt_key.startswith("moment2."):
                param_name = pt_key[len("moment2."):]
                moment_state_dict_2[param_name] = pt_tensor
            elif pt_key in ['global_step', 'learning_rate', 'beta1_power', 'beta2_power',
                            'current_iterator_step', 'last_overflow_iterator_step']:
                continue  
            else:
                torch_state_dict[pt_key] = pt_tensor
        missing_keys, unexpected_keys = self.net.load_state_dict(torch_state_dict, strict=False)
        print(f"[Load Report]")
        print(f"Missing keys in PyTorch model: {missing_keys}")
        print(f"Unexpected keys in checkpoint: {unexpected_keys}")
        self.moment_state_dict_1 = moment_state_dict_1
        self.moment_state_dict_2 = moment_state_dict_2
        # meta info
        meta_info = {}
        for key in ['global_step', 'learning_rate', 'beta1_power', 'beta2_power',
                    'current_iterator_step', 'last_overflow_iterator_step']:
            if key in self.ms_ckpt:
                val = self.ms_ckpt[key].asnumpy()
                if hasattr(val, 'shape') and val.shape == ():
                    val = val.item()
                meta_info[key] = val
        self.meta_info = meta_info

    def init_optimizer(self, lr=1e-4, weight_decay=1e-5, moment=True):
        optimizer = optim.Adam(self.params, lr=lr, weight_decay=weight_decay)
        scaler = torch.cuda.amp.GradScaler(init_scale=1.0)
        if moment:
            param_name_map = {name: param for name, param in self.net.named_parameters()}
            for name, param in param_name_map.items():
                if name in self.moment_state_dict_1 and name in self.moment_state_dict_2:
                    exp_avg = self.moment_state_dict_1[name].clone().to(param.device).to(param.dtype)
                    exp_avg_sq = self.moment_state_dict_2[name].clone().to(param.device).to(param.dtype)
                    step = torch.tensor(0, dtype=torch.float32, device='cpu')
                    optimizer.state[param]['exp_avg'] = exp_avg
                    optimizer.state[param]['exp_avg_sq'] = exp_avg_sq
                    optimizer.state[param]['step'] = step
        self.optimizer = optimizer
        self.scaler = scaler


    @staticmethod
    def restore_meta_to_torch(meta_info, optimizer):
        if 'global_step' in meta_info:
            step_tensor = torch.tensor(meta_info['global_step'], dtype=torch.float32, device='cpu')
            # restore lr
            for group in optimizer.param_groups:
                group['lr'] = meta_info.get('learning_rate', group['lr'])
            # restore step
            for param in optimizer.state:
                optimizer.state[param]['step'] = step_tensor

    def forward(self, raw_nzdata, dw_nzdata, ST_feat, nonz_gene, mask_gene, zero_idx, species_id=None, ortholog_pairs=None):
        
        if species_id is None:
            species_id = torch.zeros(raw_nzdata.size(0), dtype=torch.long, device=raw_nzdata.device)
            
        # 1. 特征拦截
        emb, gene_emb = self.net.encode(dw_nzdata, nonz_gene, ST_feat, zero_idx)
        cell_emb = emb[:, 0, :] 
            
        # 2. 核心路由 (现在会返回 4 个值，包含 specific_penalty)
        routed_gene_emb, gene_cl_loss, gene_gates, gene_spec_penalty = self.gene_moe(gene_emb, species_id)
        routed_cell_emb, cell_cl_loss, cell_gates, cell_spec_penalty = self.cell_moe(cell_emb, species_id)
        
        # 3. 特征注入 MLM 解码器
        mlm_loss = self.net(
            raw_nzdata, dw_nzdata, ST_feat, nonz_gene, mask_gene, zero_idx, 
            species_id=species_id, 
            injected_gene_emb=routed_gene_emb
        )
        
        # 4. 同源对齐损失
        base_gene_embeddings = self.net.gene_emb.base_emb if hasattr(self.net.gene_emb, 'base_emb') else self.net.gene_emb.weight
        ortho_align_loss = ortholog_alignment_loss(base_gene_embeddings, ortholog_pairs)
        
        # ==========================================
        # 5. 多任务损失函数终极融合 (包含惩罚项)
        # ==========================================
        # 设定特异性模块的惩罚权重 penalty_weight 
        # (如果发现特征还是大量涌入特异模块，可以把 2.0 调高到 5.0 甚至 10.0)
        penalty_weight = 2.0 
        
        total_loss = mlm_loss + \
                     (0.1 * gene_cl_loss) + (0.1 * cell_cl_loss) + \
                     (0.5 * ortho_align_loss) + \
                     (penalty_weight * gene_spec_penalty) + (penalty_weight * cell_spec_penalty)
        
        # 注意：gene_gates 现在的 shape 变成了 [Batch, Seq_len, 8]
        return total_loss, routed_cell_emb, gene_gates, cell_gates
    
class Finetune_Cell_FM(nn.Module):
    def __init__(self, cfg):
        super(Finetune_Cell_FM, self).__init__()
        self.cfg = cfg
        self.num_cls = cfg.num_cls
        self.extractor = Cell_FM(self.cfg.n_genes, self.cfg, ckpt_path=self.cfg.ckpt_path, device=self.cfg.device)
        #self.extractor = Cell_FM(27855, self.cfg, ckpt_path=self.cfg.ckpt_path, device=self.cfg.device) # n_gene, cfg=config_80M()
        self.cls = nn.Sequential(
            nn.Linear(self.cfg.enc_dims, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, self.num_cls)
        )
    
    def forward(self, raw_nzdata,
                dw_nzdata,
                ST_feat,
                nonz_gene,
                mask_gene,
                zero_idx):
        
        mask_loss, cls_token = self.extractor(
                raw_nzdata,
                dw_nzdata,
                ST_feat,
                nonz_gene,
                mask_gene,
                zero_idx
            )
        
        cls = self.cls(cls_token)

        return cls, mask_loss, cls_token
