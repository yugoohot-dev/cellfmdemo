import scanpy as sc
import os
import torch
import torch.nn as nn
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
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
from model import Cell_FM

# ==========================================
# 0. 自定义跨物种数据加载器 (绕过底层的硬编码基因过滤)
# ==========================================
class CrossSpeciesSCrna(Dataset):
    def __init__(self, adata, global_species=0): # 增加 global_species 参数
        self.adata = adata
        if not sp.issparse(self.adata.X):
            self.adata.X = sp.csr_matrix(self.adata.X)
        self.T = np.asarray(self.adata.X.sum(1)).ravel()
        self.gene = np.arange(self.adata.n_vars, dtype=np.int32)

        # 直接使用外部传进来的 global_species
        if 'species_id' in adata.obs.columns:
            self.species_ids = adata.obs['species_id'].values
        else:
            self.species_ids = np.full(adata.n_obs, global_species)

    def __len__(self):
        return self.adata.n_obs

    def __getitem__(self, idx):
        data = self.adata.X[idx].toarray().ravel().astype(np.float32)
        T = np.asarray(self.T[idx], dtype=np.float32)
        species_id = int(self.species_ids[idx])
        return data, self.gene, T, species_id, 0, 0.0
  
# ==========================================
# 1. 跨物种基因对齐  Symbol->ID 翻译)
# ==========================================
def align_cross_species_adata(adata, token_dict, base_vocab_size, symbol_to_id):
    print(f"Original adata shape: {adata.shape}")
    X_orig = adata.X.tocsc() if sp.issparse(adata.X) else sp.csc_matrix(adata.X)
    new_X = sp.lil_matrix((adata.n_obs, base_vocab_size), dtype=np.float32)
    print("词表维度")
    print(new_X.shape)
    
    matched = 0
    for i, gene in enumerate(adata.var_names):
        g_upper = str(gene).upper()
        ensembl_id = symbol_to_id.get(g_upper, g_upper)
        if ensembl_id in token_dict:
            token_id = token_dict[ensembl_id]
            new_X[:, token_id] = X_orig[:, i]
            matched += 1
            
    print(f"Data Alignment: Matched {matched} out of {adata.n_vars} genes to unified vocabulary.")
    print(new_X.shape)
    print(new_X)
    new_adata = sc.AnnData(X=new_X.tocsr(), obs=adata.obs)
    new_adata.var_names = [str(i) for i in range(base_vocab_size)] 
    return new_adata

# 1. 重构先验知识矩阵 (仅包含 Promoter + Family)
# ==========================================
def build_prior_knowledge_matrix(token_dict, prior_dir, vocab_size, id_to_name):
    print("Building Universal Prior Knowledge Matrix (Promoter 768 + Gene Family 768)...")
    prior_matrix = torch.zeros(vocab_size, 1536) # 1536维
    
    def load_pk(filename):
        path = os.path.join(prior_dir, filename)
        return pickle.load(open(path, 'rb')) if os.path.exists(path) else {}

    # 加载你的 6 大物种启动子与家族先验
    prom_all = {
        **load_pk("human_promoter_emb_768.pickle"), **load_pk("mouse_promoter_emb_768.pickle"),
        **load_pk("zebrafish_promoter_emb_768.pickle"), **load_pk("chicken_promoter_emb_768.pickle"),
        **load_pk("frog_promoter_emb_768.pickle"), **load_pk("macaque_promoter_emb_768.pickle")
    }
    fam_all = load_pk("universal_gene_family_emb_768.pickle")
    
    
    


    # 建立反向字典 Symbol -> ENSG
    name_to_id = {str(v).upper(): k for k, v in id_to_name.items()}
    
    for gene_key, token_id in token_dict.items():
        if token_id >= vocab_size: continue
        
        # 1. 拆出前缀和纯净符号 (如: gene_key="MAC_GAPDH" -> prefix="MAC_", pure_symbol="GAPDH")
        if '_' in gene_key:
            prefix, pure_symbol = gene_key.split('_', 1)
            prefix = prefix + '_'
        else:
            prefix = ""
            pure_symbol = gene_key
            
        # 2. 映射回原始 ID（如果是 Symbol 就转为 ENSG，否则保持原样）
        raw_id = name_to_id.get(pure_symbol, pure_symbol)
        
        # 3. 带上前缀，去你生成的超级启动子字典里精准打捞特征！
        query_key = f"{prefix}{raw_id}"
        
        emb_prom = prom_all.get(query_key, torch.randn(768) * 0.02)
        emb_fam  = fam_all.get(query_key, torch.randn(768) * 0.02)


        
        if not isinstance(emb_prom, torch.Tensor): emb_prom = torch.tensor(emb_prom)
        if not isinstance(emb_fam, torch.Tensor): emb_fam = torch.tensor(emb_fam)
            
        prior_matrix[token_id] = torch.cat([emb_prom.float(), emb_fam.float()])
        
    print("✅ Prior Matrix Built!")
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
# 同源权重克隆外科手术与断点续训管理器
# ==========================================
class CrossSpecies_Cell_FM(Cell_FM):
    def __init__(self, n_gene, cfg, ckpt_path=None, device=None, token_dict=None, symbol_to_id=None, ortholog_dict=None):
        super().__init__(n_gene, cfg, ckpt_path, device)
        self.token_dict = token_dict
        self.symbol_to_id = symbol_to_id
        self.ortholog_dict = ortholog_dict # 格式：{非人基因ID: 人类同源ID}
        
    def load_weight_and_surgery(self, prior_matrix):
        import mindspore as ms
        
        # 1. 无论哪种情况，必须先挂载 1536 维先验特征外挂骨架 (组装好网络结构)
        orig_tensor = self.net.gene_emb.data.clone()
        del self.net.gene_emb
        full_prior = torch.zeros((orig_tensor.shape[0], prior_matrix.shape[1]))
        full_prior[:prior_matrix.shape[0], :] = prior_matrix
        
        # 挂载拦截式先验
        self.net.gene_emb = PriorAugmentedEmbedding(
            orig_tensor, full_prior.to(self.cfg.device), self.cfg.enc_dims
        ).to(self.cfg.device)

        if not self.ckpt_path: 
            print("No checkpoint provided. Training from scratch.")
            return

        # ==========================================
        # 场景 A: 首次训练，加载 MindSpore 预训练底座进行外科手术
        # ==========================================
        if self.ckpt_path.endswith('.ckpt'):
            print(f"Loading Base Checkpoint from {self.ckpt_path} ...")
            self.ms_ckpt = ms.load_checkpoint(self.ckpt_path)
            torch_state_dict = {self.map_ms_to_pt(k): torch.tensor(v.asnumpy()) 
                                for k, v in self.ms_ckpt.items() if not self.map_ms_to_pt(k).startswith("moment")}
                    
            if "gene_emb" in torch_state_dict:
                old_emb = torch_state_dict["gene_emb"]
                new_emb = self.net.gene_emb.base_emb.data 
                
                # 步骤 A：复原人类基础基因的权重
                human_id_to_old_idx = {}
                csv_path = 'csv/expand_gene_info.csv'
                if os.path.exists(csv_path):
                    import pandas as pd
                    cellfm_gene_info = pd.read_csv(csv_path, index_col=0, header=0)
                    for i, cellfm_symbol in enumerate(cellfm_gene_info.index):
                        cellfm_id = i + 1
                        if cellfm_id >= old_emb.shape[0]: continue
                        ensembl_id = self.symbol_to_id.get(str(cellfm_symbol).upper(), None)
                        
                        if ensembl_id and ensembl_id in self.token_dict:
                            new_token_id = self.token_dict[ensembl_id]
                            if new_token_id < new_emb.shape[0]:
                                new_emb[new_token_id, :] = old_emb[cellfm_id, :]
                                human_id_to_old_idx[ensembl_id] = cellfm_id
                                
                # 步骤 B：同源基因权重克隆 (Homologous Surgery)
                surgery_count = 0
                if self.ortholog_dict:
                    for other_id, human_id in self.ortholog_dict.items():
                        if other_id in self.token_dict and human_id in human_id_to_old_idx:
                            new_idx = self.token_dict[other_id]
                            old_idx = human_id_to_old_idx[human_id]
                            # 直接克隆人类的参数作为该物种同源基因的初始化点
                            new_emb[new_idx, :] = old_emb[old_idx, :]
                            surgery_count += 1
                    print(f"🧬 Homologous Surgery: Safely cloned weights for {surgery_count} ortholog genes!")

                del torch_state_dict["gene_emb"]

            # 【关键区别】：如果是 ckpt，我们只用它初始化 backbone (self.net)
            # MoE 和 对比学习的参数此时是干净的随机初始化状态
            self.net.load_state_dict(torch_state_dict, strict=False)
            print("✅ [Init] Base model loaded and ortholog surgery completed.")

        # ==========================================
        # 场景 B: 断点续训，加载 PyTorch (.pth) 恢复训练
        # ==========================================
        elif self.ckpt_path.endswith('.pth'):
            print(f"🚀 Resuming continual training from checkpoint: {self.ckpt_path} ...")
            pt_state_dict = torch.load(self.ckpt_path, map_location='cpu')
            
            # 【关键区别】：使用 self.load_state_dict 直接加载整个大模型！
            # 这不仅会恢复微调过的特征，还会完美恢复 gene_moe 和 cell_moe 的路由权重。
            # strict=False 用于忽略由于固定 prior_matrix 导致的多余/缺失 key 警告。
            self.load_state_dict(pt_state_dict, strict=False)
            
            print("✅ [Resume] Model states (including MoE routers & tuned embeddings) fully restored.")

# ==========================================
# 5. 主预训练流程
# ==========================================
def pretrain(args):
    cfg = Config_80M()
    cfg.ecs = False
    cfg.ecs_threshold = 0.8
    cfg.add_zero = False
    cfg.pad_zero = True
    
    cfg.use_bs = args.batch_size
    cfg.mask_ratio = 0.20 
    cfg.ckpt_path = args.ckpt_path
    cfg.device = args.device
    cfg.epoch = args.epoch 
    
    with open(args.token_dict_path, 'rb') as f:
        token_dict = pickle.load(f)
    print(type(token_dict))
    print(len(token_dict.items()))
    name_dict_path = os.path.join(args.prior_dir, "gene_list", "Gene_id_name_dict_human_mouse.pickle")
    with open(name_dict_path, 'rb') as f:
        id_to_name = pickle.load(f)
    print("Building Species-Specific & Human Global Mappings...")
    is_human = "human" in args.data_path.lower()
    
    symbol_to_id = {}          # 专用于当前数据集对齐的 Symbol -> 独立物种 Ensembl 映射
    symbol_to_ens_human = {}   # 专用于基座权重定位的 Symbol -> 人类 ENSG 映射
    
    for ens_id, name in id_to_name.items():
        g_upper = str(name).upper()
        ens_str = str(ens_id)
        
        # 1. 建立基准人类参考系（为了上面的权重寻找）
        if ens_str.startswith("ENSG"):
            symbol_to_ens_human[g_upper] = ens_str
            
        # 2. 建立物种隔离的数据集映射器 (彻底保留跨物种差异，不发生人类强制覆盖)
        if is_human and ens_str.startswith("ENSG"):
            symbol_to_id[g_upper] = ens_str
        elif not is_human and not ens_str.startswith("ENSG"):
            # 匹配 ENSMUSG, ENSDARG 等
            symbol_to_id[g_upper] = ens_str

    print(f"Successfully built mapping for {len(symbol_to_id)} dataset-specific genes.")

    # 3. 加载所有非人物种到人类的同源字典 (用于权重传染)
    homology_dict = {}
    homo_path = "mouse2human_ensembl.pkl" # 这里利用了你已有的文件
    if os.path.exists(homo_path):
        with open(homo_path, 'rb') as f:
            homology_dict = pickle.load(f)
        print(f"Loaded {len(homology_dict)} homologous pairs for zero-shot weight transfer.")

    base_vocab_size = len(token_dict)
    cfg.n_genes = base_vocab_size
    print(f"Total Vocabulary Size (including 2 Species Tokens): {cfg.n_genes}")
    
    # ... 后续加载数据adata逻辑不变 ...
    
    
    MODEL_PATH = f"../model_checkpoint/immune_multispecies_pretrain"
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    print(f"Loading data from {args.data_path}...")
    adata = read_h5ad(args.data_path)
    adata = align_cross_species_adata(adata, token_dict, base_vocab_size, symbol_to_id)
    
    # [核心修复] 使用自定义数据集类，保留全长跨物种特征矩阵
    # 提前判定是人还是鼠
    is_human = "human" in args.data_path.lower()
    global_species_val = 0 if is_human else 1

    # 传给 Dataset
    dataset = CrossSpeciesSCrna(adata, global_species=global_species_val)
    
    prep = Prepare(cfg.nonz_len, pad=0, mask_ratio=cfg.mask_ratio)
    train_loader = build_dataset(dataset, prep=prep, batch_size=cfg.use_bs, pad_zero=cfg.pad_zero, drop=True, shuffle=True)
    
    prior_matrix = build_prior_knowledge_matrix(token_dict, args.prior_dir, cfg.n_genes, id_to_name)
    
    # 初始化模型时传入字典
    # 初始化模型时，将新的字典全部注入进去
    net = CrossSpecies_Cell_FM(
        cfg.n_genes, cfg, 
        ckpt_path=cfg.ckpt_path, 
        device=cfg.device, 
        token_dict=token_dict, 
        symbol_to_id=symbol_to_id,               # 数据对齐映射
        symbol_to_ens_human=symbol_to_ens_human, # 权重定位锚点
        homology_dict=homology_dict              # 一对一同源投射桥梁
    )
    net.load_weight_and_surgery(prior_matrix)  
    
    net = net.to(cfg.device)
    
    # 静态先验知识矩阵不被解冻参与梯度计算
    for name, param in net.named_parameters():
        if "prior_matrix" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    optimizer = optim.AdamW(net.parameters(), lr=1e-5, weight_decay=1e-4)
    scaler = GradScaler() 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epoch) 
    
    is_human = "human" in args.data_path.lower() 
    if 'species_id' not in adata.obs.columns:
        adata.obs['species_id'] = 0 if is_human else 1

    # [新增] 收集 Loss 的列表
    step_losses = []

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
            # ====================================================
            # # 从 batch 中提取 species_id
            species_id = batch['celltype_label'].to(cfg.device)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                # 直接传 species_id 给网络
                loss, _ = net(
                    raw_nzdata=raw_nzdata,
                    dw_nzdata=dw_nzdata,
                    ST_feat=ST_feat,
                    nonz_gene=nonz_gene,
                    mask_gene=mask_gene,
                    zero_idx=zero_idx,
                    species_id=species_id
                ) 
            
            scaler.scale(loss).backward()
            
            # 🛑 修复 AMP 梯度裁剪致命 Bug：必须先 unscale！
            scaler.unscale_(optimizer) 
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            loss_val = loss.item()
            step_losses.append(loss_val)
            
            running_loss += loss_val
            progress.set_postfix(Pretrain_Loss=running_loss/(step+1))
        
        scheduler.step()
        save_file = f"{MODEL_PATH}/cellfm_immune_multispecies_epoch_{epoch+1}.pth"
        torch.save(net.state_dict(), save_file)
        print(f"Model saved: {save_file}")

    # ==========================================
    # [新增] 绘制单条 Total Loss 曲线图
    # ==========================================
    print("Generating Total Loss curve...")
    plt.figure(figsize=(10, 6))
    
    plt.plot(step_losses, alpha=0.3, color='#1f77b4', label='Step Total Loss')
    
    window_size = 50
    if len(step_losses) > window_size:
        smoothed_loss = np.convolve(step_losses, np.ones(window_size)/window_size, mode='valid')
        plt.plot(np.arange(window_size-1, len(step_losses)), smoothed_loss, color='#d62728', linewidth=2, label=f'Smoothed Loss (Window={window_size})')

    plt.title('Continual Pre-training Total Loss', fontsize=14)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plot_save_path = os.path.join(MODEL_PATH, 'pretrain_total_loss_curve.png')
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Total Loss curve plot saved successfully to: {plot_save_path}")

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
