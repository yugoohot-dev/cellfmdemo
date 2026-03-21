import scanpy as sc
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset
from layers.utils import *
import numpy as np
import scipy.sparse as sp
import argparse
from tqdm import tqdm
import pickle
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
from model import Cell_FM

# ==========================================
# 0. 自定义跨物种数据加载器 (绕过底层的硬编码基因过滤)
# ==========================================
class CrossSpeciesSCrna(Dataset):
    def __init__(self, adata):
        self.adata = adata
        # 确保使用 CSR 稀疏矩阵加速按行读取
        if not sp.issparse(self.adata.X):
            self.adata.X = sp.csr_matrix(self.adata.X)
        self.T = np.asarray(self.adata.X.sum(1)).ravel()
        # [关键] 直接使用我们的跨物种 Token ID (0 到 50557) 作为特征索引
        self.gene = np.arange(self.adata.n_vars, dtype=np.int32)

    def __len__(self):
        return self.adata.n_obs

    def __getitem__(self, idx):
        # 提取细胞的全量 5 万维特征交由底层的 Prepare 类去动态截断/遮盖
        data = self.adata.X[idx].toarray().ravel().astype(np.float32)
        T = np.asarray(self.T[idx], dtype=np.float32)
        # 占位下游不用的分类 label 和 batch
        return data, self.gene, T, 0, 0, 0.0

# ==========================================
# 1. 跨物种基因对齐 (附带 Symbol->ID 翻译)
# ==========================================
def align_cross_species_adata(adata, token_dict, base_vocab_size, symbol_to_id):
    print(f"Original adata shape: {adata.shape}")
    X_orig = adata.X.tocsc() if sp.issparse(adata.X) else sp.csc_matrix(adata.X)
    new_X = sp.lil_matrix((adata.n_obs, base_vocab_size), dtype=np.float32)
    
    matched = 0
    for i, gene in enumerate(adata.var_names):
        g_upper = str(gene).upper()
        ensembl_id = symbol_to_id.get(g_upper, g_upper)
        if ensembl_id in token_dict:
            token_id = token_dict[ensembl_id]
            new_X[:, token_id] = X_orig[:, i]
            matched += 1
            
    print(f"Data Alignment: Matched {matched} out of {adata.n_vars} genes to unified vocabulary.")
    new_adata = sc.AnnData(X=new_X.tocsr(), obs=adata.obs)
    new_adata.var_names = [str(i) for i in range(base_vocab_size)] 
    return new_adata

# ==========================================
# 2. 先验知识矩阵构建
# ==========================================
def build_prior_knowledge_matrix(token_dict, prior_dir, vocab_size, id_to_name):
    print("Building Prior Knowledge Matrix...")
    prior_dim = 768 * 4
    prior_matrix = torch.zeros(vocab_size, prior_dim)
    
    def load_pk(path):
        full_path = os.path.join(prior_dir, path)
        if not os.path.exists(full_path):
            return {}
        with open(full_path, 'rb') as f:
            return pickle.load(f)

    h_peca = load_pk("PECA2vec/human_PECA_vec.pickle")
    m_peca = load_pk("PECA2vec/mouse_PECA_vec.pickle")
    h_prom = load_pk("promoter_emb/human_emb_768.pickle")
    m_prom = load_pk("promoter_emb/mouse_emb_768.pickle")
    h_fam = load_pk("gene_family/Human_dim_768_gene_28291_random.pickle")
    m_fam = load_pk("gene_family/Mouse_dim_768_gene_27934_random.pickle")
    h_co = load_pk("gene_co_express_emb/Human_dim_768_gene_28291_random.pickle")
    m_co = load_pk("gene_co_express_emb/Mouse_dim_768_gene_27444_random.pickle")
    
    peca_all = {**h_peca, **m_peca}
    prom_all = {**h_prom, **m_prom}
    fam_all = {**h_fam, **m_fam}
    co_all = {**h_co, **m_co}
    
    matched = 0
    for gene_id, token_id in token_dict.items():
        if token_id >= vocab_size: continue
        
        symbol = str(id_to_name.get(gene_id, ""))
        candidates = [gene_id, symbol, symbol.upper(), symbol.capitalize()]
        
        def get_emb(pk_dict):
            for k in candidates:
                if k and k in pk_dict:
                    val = pk_dict[k]
                    if isinstance(val, torch.Tensor): return val.clone().detach().float()
                    else: return torch.tensor(val, dtype=torch.float32)
            return torch.zeros(768, dtype=torch.float32)
        
        emb_peca = get_emb(peca_all)
        emb_prom = get_emb(prom_all)
        emb_fam  = get_emb(fam_all)
        emb_co   = get_emb(co_all)
        
        cat_emb = torch.cat([emb_peca, emb_prom, emb_fam, emb_co])
        if cat_emb.abs().sum() > 0:
            matched += 1
        prior_matrix[token_id] = cat_emb
        
    print(f"Prior Knowledge Builder: Successfully matched features for {matched} genes.")
    return prior_matrix

# ==========================================
# 3. 拦截式先验知识融合包装器
# ==========================================
class PriorAugmentedEmbedding(nn.Module):
    def __init__(self, orig_tensor, prior_matrix, enc_dims):
        super().__init__()
        self.base_emb = nn.Parameter(orig_tensor)
        self.prior_matrix = nn.Parameter(prior_matrix, requires_grad=False)
        self.prior_proj = nn.Linear(prior_matrix.shape[1], enc_dims)
        self.ln = nn.LayerNorm(enc_dims)

    def __getitem__(self, idx):
        base = self.base_emb[idx]
        prior = self.prior_proj(self.prior_matrix[idx])
        return self.ln(base + prior)

# ==========================================
# 4. 扩容与外挂融合模型
# ==========================================
class CrossSpecies_Cell_FM(Cell_FM):
    def __init__(self, n_gene, cfg, ckpt_path=None, device=None):
        super().__init__(n_gene, cfg, ckpt_path, device)
        
    def load_weight_and_surgery(self, prior_matrix):
        import mindspore as ms
        print(f"Loading base checkpoint from {self.ckpt_path} ...")
        self.ms_ckpt = ms.load_checkpoint(self.ckpt_path)
        
        torch_state_dict = {}
        for ms_key, ms_param in self.ms_ckpt.items():
            pt_key = self.map_ms_to_pt(ms_key)
            if not pt_key.startswith("moment") and pt_key not in ['global_step', 'learning_rate']:
                torch_state_dict[pt_key] = torch.tensor(ms_param.asnumpy())
                
        if "gene_emb" in torch_state_dict:
            old_emb = torch_state_dict["gene_emb"]
            new_emb = self.net.gene_emb.data 
            copy_len = min(old_emb.shape[0], new_emb.shape[0])
            new_emb[:copy_len, :] = old_emb[:copy_len, :]
            print(f"[Surgery] Transferred {copy_len} base gene embeddings.")
            del torch_state_dict["gene_emb"]

        self.net.load_state_dict(torch_state_dict, strict=False)
        
        orig_tensor = self.net.gene_emb.data.clone()
        del self.net.gene_emb
        
        self.net.gene_emb = PriorAugmentedEmbedding(
            orig_tensor, 
            prior_matrix.to(self.cfg.device), 
            self.cfg.enc_dims
        ).to(self.cfg.device)
        print("[Surgery] Prior Knowledge mapping network successfully attached.")

    def forward(self, raw_nzdata, dw_nzdata, ST_feat, nonz_gene, mask_gene, zero_idx):
        emb, gene_emb = self.net.encode(dw_nzdata, nonz_gene, ST_feat, zero_idx)
        cls_token, st_emb, expr_emb = emb[:, 0], emb[:, 1:3], emb[:, 3:]

        if self.net.add_zero:
            gw_pred, z_prob1 = self.net.value_dec(expr_emb)
            cw_pred, z_prob2 = self.net.cellwise_dec(cls_token, gene_emb)
        else:
            gw_pred = self.net.value_dec(expr_emb)
            cw_pred = self.net.cellwise_dec(cls_token, gene_emb)

        loss1 = self.net.reconstruct1(gw_pred, raw_nzdata, mask_gene)
        loss2 = self.net.reconstruct2(cw_pred, raw_nzdata, mask_gene)
        total_loss = loss1 + loss2
        
        loss_dict = {
            'MSE_Global': loss1,
            'MSE_Cellwise': loss2,
            'BCE_Global': torch.tensor(0.0, device=self.device),
            'BCE_Cellwise': torch.tensor(0.0, device=self.device),
            'ECS_Regularization': torch.tensor(0.0, device=self.device)
        }

        if self.net.training and self.net.add_zero:
            nonz_pos = zero_idx
            loss3 = self.net.bce_loss1(z_prob1, nonz_pos, mask_gene)
            loss4 = self.net.bce_loss2(z_prob2, nonz_pos, mask_gene)
            total_loss = total_loss + loss3 + loss4
            loss_dict['BCE_Global'] = loss3
            loss_dict['BCE_Cellwise'] = loss4

        if self.net.training and self.cfg.ecs:
            cell_emb_normed = F.normalize(cw_pred, p=2, dim=1)
            cos_sim = torch.matmul(cell_emb_normed, cell_emb_normed.transpose(0, 1))
            eye_mask = torch.eye(cos_sim.size(0), device=cos_sim.device, dtype=cos_sim.dtype)
            cos_sim = cos_sim * (1 - eye_mask)
            cos_sim = F.relu(cos_sim)
            loss5 = torch.mean(1 - (cos_sim - self.net.ecs_threshold) ** 2)
            total_loss += loss5
            loss_dict['ECS_Regularization'] = loss5

        loss_dict['Total_Loss'] = total_loss
        return total_loss, cls_token, loss_dict

# ==========================================
# 5. 主预训练流程
# ==========================================
def pretrain(args):
    cfg = Config_80M()
    cfg.ecs = True
    cfg.ecs_threshold = 0.8
    cfg.add_zero = True
    cfg.pad_zero = False
    
    cfg.use_bs = args.batch_size
    cfg.mask_ratio = 0.20 
    cfg.ckpt_path = args.ckpt_path
    cfg.device = args.device
    cfg.epoch = args.epoch 
    
    with open(args.token_dict_path, 'rb') as f:
        token_dict = pickle.load(f)
        
    name_dict_path = os.path.join(args.prior_dir, "gene_list", "Gene_id_name_dict_human_mouse.pickle")
    with open(name_dict_path, 'rb') as f:
        id_to_name = pickle.load(f)
    symbol_to_id = {str(name).upper(): str(ens_id) for ens_id, name in id_to_name.items()}
    
    base_vocab_size = len(token_dict)
    HUMAN_TOKEN_ID = base_vocab_size 
    MOUSE_TOKEN_ID = base_vocab_size + 1
    cfg.n_genes = base_vocab_size + 2 
    print(f"Total Vocabulary Size (including 2 Species Tokens): {cfg.n_genes}")
    
    MODEL_PATH = f"../model_checkpoint/immune_multispecies_pretrain"
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    print(f"Loading data from {args.data_path}...")
    adata = read_h5ad(args.data_path)
    adata = align_cross_species_adata(adata, token_dict, base_vocab_size, symbol_to_id)
    
    # [核心修复]: 使用我们强大的自定义数据集类，避开底层残次逻辑
    dataset = CrossSpeciesSCrna(adata)
    
    prep = Prepare(cfg.nonz_len, pad=0, mask_ratio=cfg.mask_ratio)
    train_loader = build_dataset(dataset, prep=prep, batch_size=cfg.use_bs, pad_zero=cfg.pad_zero, drop=True, shuffle=True)
    
    prior_matrix = build_prior_knowledge_matrix(token_dict, args.prior_dir, cfg.n_genes, id_to_name)
    
    net = CrossSpecies_Cell_FM(cfg.n_genes, cfg, ckpt_path=cfg.ckpt_path, device=cfg.device) 
    net.load_weight_and_surgery(prior_matrix)  
    net = net.to(cfg.device)
    
    for name, param in net.named_parameters():
        if "prior_matrix" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    optimizer = optim.AdamW(net.parameters(), lr=1e-5, weight_decay=1e-4)
    scaler = GradScaler() 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epoch) 
    
    is_human = "human" in args.data_path.lower() 

    loss_history = {
        'Total_Loss': [],
        'MSE_Global': [],
        'MSE_Cellwise': [],
        'BCE_Global': [],
        'BCE_Cellwise': [],
        'ECS_Regularization': []
    }

    for epoch in range(cfg.epoch):
        net.train()
        print(f"--- Continual Pre-training Epoch {epoch+1}/{cfg.epoch} ---")
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(progress):    
            raw_nzdata = batch['raw_nzdata'].to(cfg.device)
            dw_nzdata = batch['dw_nzdata'].to(cfg.device)
            ST_feat = batch['ST_feat'].to(cfg.device)
            nonz_gene = batch['nonz_gene'].to(cfg.device)
            mask_gene = batch['mask_gene'].to(cfg.device)
            zero_idx = batch['zero_idx'].to(cfg.device)

            # ==========================================================
            # [核心修复]：所有序列级 Tensor 必须同步平移，绝不能错位！
            # ==========================================================
            # 1. 平移输入序列 (Input)
            nonz_gene[:, 1:] = nonz_gene[:, :-1].clone()
            dw_nzdata[:, 1:] = dw_nzdata[:, :-1].clone()
            nonz_gene[:, 0] = HUMAN_TOKEN_ID if is_human else MOUSE_TOKEN_ID
            dw_nzdata[:, 0] = 1.0 
            
            # 2. 平移目标答案 (Ground Truth)
            raw_nzdata[:, 1:] = raw_nzdata[:, :-1].clone()
            raw_nzdata[:, 0] = 1.0  # 设为1.0即可，因为下方的 mask 会把它屏蔽掉
            
            # 3. 平移损失计算掩码 (Loss Mask)
            mask_gene[:, 1:] = mask_gene[:, :-1].clone()
            mask_gene[:, 0] = 0.0   # [关键] 告诉模型：不要对第0位的物种Token算MSE误差！
            
            # 4. 平移有效长度掩码 (Attention Pad Mask)
            zero_idx[:, 1:] = zero_idx[:, :-1].clone()
            zero_idx[:, 0] = 1.0    # 第0位是真实的物种Token，不是为了凑齐2048补的0(Pad)
            # ==========================================================

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss, cls_token, loss_dict = net(
                    raw_nzdata=raw_nzdata,
                    dw_nzdata=dw_nzdata,
                    ST_feat=ST_feat,
                    nonz_gene=nonz_gene,
                    mask_gene=mask_gene,
                    zero_idx=zero_idx
                ) 
            
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            for k, v in loss_dict.items():
                loss_history[k].append(v.item())
            
            running_loss += loss.item()
            progress.set_postfix(Total_Loss=running_loss/(step+1))
        
        scheduler.step()
        save_file = f"{MODEL_PATH}/cellfm_immune_multispecies_epoch_{epoch+1}.pth"
        torch.save(net.state_dict(), save_file)
        print(f"Model saved: {save_file}")

    print("Generating comprehensive loss curve grid...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Continual Pre-training Loss Components', fontsize=18)
    axes = axes.flatten()

    titles = ['Total_Loss', 'MSE_Global', 'MSE_Cellwise', 'BCE_Global', 'BCE_Cellwise', 'ECS_Regularization']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    window_size = 50 
    for i, key in enumerate(titles):
        ax = axes[i]
        y_vals = loss_history[key]
        
        ax.plot(y_vals, alpha=0.3, color=colors[i])
        
        if len(y_vals) > window_size:
            smoothed = np.convolve(y_vals, np.ones(window_size)/window_size, mode='valid')
            x_vals = np.arange(window_size-1, len(y_vals))
            ax.plot(x_vals, smoothed, color=colors[i], linewidth=2, label=f'Smoothed (W={window_size})')
            
        ax.set_title(key, fontsize=14)
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Loss Value')
        ax.grid(True, linestyle='--', alpha=0.6)
        if len(y_vals) > window_size:
            ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_save_path = os.path.join(MODEL_PATH, 'pretrain_losses_grid.png')
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"All individual loss curves successfully saved to: {plot_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--token_dict_path", type=str, required=True)
    parser.add_argument("--prior_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--device", type=str, default='cuda:0')
    args = parser.parse_args()
    pretrain(args)