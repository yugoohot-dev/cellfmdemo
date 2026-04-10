import numpy as np
from .torch_retention import *
import torch.nn.init as init
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 损失函数引擎：邻居对比学习 & 同源对齐
# ==========================================
def neighbor_contrastive_loss(features, temperature=0.1, top_k=3):
    """ 
    邻居对比学习：不光和自己对比，也将当前 Batch 内余弦相似度最高的 top_k 个样本视为正样本（邻居），拉近相似细胞/基因的距离。
    """
    features = F.normalize(features, dim=-1)
    # 计算相似度矩阵 [N, N]
    sim_matrix = torch.matmul(features, features.T) / temperature
    
    # 构造邻居掩码 (将 top_k 个最相似的样本视为正样本)
    with torch.no_grad():
        _, top_k_indices = torch.topk(sim_matrix, k=top_k+1, dim=-1) # +1 包含自己
        positive_mask = torch.zeros_like(sim_matrix).scatter_(1, top_k_indices, 1.0)
    
    # 计算 InfoNCE 变体 (基于正样本集合)
    # log_prob = log( exp(sim) / sum(exp(sim)) )
    log_prob = F.log_softmax(sim_matrix, dim=-1)
    loss = - (positive_mask * log_prob).sum(dim=-1) / positive_mask.sum(dim=-1)
    return loss.mean()

def ortholog_alignment_loss(gene_emb, ortholog_pairs_in_batch):
    """ 
    同源基因对齐损失：强制批次内配对的同源基因特征保持极高的余弦相似度
    ortholog_pairs_in_batch: shape [N, 2], 表示 (非人基因ID_idx, 对应的人类基因ID_idx)
    """
    if ortholog_pairs_in_batch is None or len(ortholog_pairs_in_batch) == 0:
        return torch.tensor(0.0).to(gene_emb.device)
    
    # 假设 gene_emb 是 [Batch, Vocab_Size, Dim] 或 [Vocab_Size, Dim]
    # 这里我们针对整体基因词表特征进行对齐
    emb_A = gene_emb[ortholog_pairs_in_batch[:, 0], :]
    emb_B = gene_emb[ortholog_pairs_in_batch[:, 1], :]
    
    cos_sim = F.cosine_similarity(emb_A, emb_B, dim=-1)
    return (1.0 - cos_sim).mean() # 最小化距离，最大化相似度

# ==========================================
# 2. 物种感知混合专家网络 (Species-Aware MoE)
# ==========================================
class SpeciesAwareMoE(nn.Module):
    def __init__(self, dim, num_species=6, num_shared_experts=8, is_cell_level=False):
        super().__init__()
        self.is_cell_level = is_cell_level
        self.num_species = num_species
        self.num_shared = num_shared_experts
        
        self.species_embedding = nn.Embedding(num_species, 64)
        
        # 【修改点 1】：路由器输出维度变为 8(共享) + 1(特异) = 9
        self.router = nn.Sequential(
            nn.Linear(dim + 64, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_shared + 1)
        )
        
        # 【修改点 2】：8 个平行的共享专家，捕捉不同的保守表达模式
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim)
            ) for _ in range(self.num_shared)
        ])
        
        # 【保持】：6 个特化专家，每个物种专属 1 个
        self.specific_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim)
            ) for _ in range(num_species)
        ])

    def forward(self, x, species_id):
        b = x.size(0)
        s_emb = self.species_embedding(species_id.long().squeeze(-1) if species_id.dim() > 1 else species_id.long())
        if not self.is_cell_level:
            s_emb = s_emb.unsqueeze(1).expand(-1, x.size(1), -1)
            
        # 1. 动态路由计算 (9 维概率分布)
        gates = F.softmax(self.router(torch.cat([x, s_emb], dim=-1)), dim=-1)
        
        # 拆分门控权重：前 8 个给共享专家，最后 1 个给特化专家
        shared_gates = gates[..., :self.num_shared]    # [..., 8]
        specific_gate = gates[..., self.num_shared:]   # [..., 1]
        
        # 2. 计算并融合 8 个共享专家的输出
        shared_mix = torch.zeros_like(x)
        for i in range(self.num_shared):
            # 将第 i 个共享专家的特征，乘以其对应的分配权重
            shared_mix += shared_gates[..., i:i+1] * self.shared_experts[i](x)
            
        # 3. 计算特化专家输出 (严格的 1 对 1 物种限制)
        specific_out = torch.zeros_like(x)
        for i in range(b):
            sp_idx = int(species_id[i].item())
            if sp_idx >= self.num_species: sp_idx = 0
            specific_out[i] = self.specific_experts[sp_idx](x[i])
                
        # 4. 最终特征融合
        out = shared_mix + (specific_gate * specific_out)
        
        # ==========================================
        # 5. 【新增】：计算分配惩罚项 (Penalty Loss)
        # 惩罚 specific_gate 的平均激活值，防止网络偷懒把数据全扔给特异模块
        # ==========================================
        specific_penalty = specific_gate.mean()
        
        # 6. 执行共享专家内的邻居对比学习
        cl_loss = torch.tensor(0.0).to(x.device)
        total_shared_gate = shared_gates.sum(dim=-1, keepdim=True) # 计算该 token 分配给所有共享专家的总权重
        
        if self.training:
            # 只有当总共享权重 > 0.5 时，才认为它是保守特征，送入对比学习
            mask = (total_shared_gate > 0.5).squeeze(-1)
            if mask.sum() > 4:
                # 提取纯粹的共享特征，排除特化干扰
                pure_shared_features = shared_mix[mask] / total_shared_gate[mask]
                cl_loss = neighbor_contrastive_loss(pure_shared_features, top_k=3)
                
        # 返回：融合特征、对比损失、共享门控分布(用于下游量化)、特异惩罚损失
        return out, cl_loss, shared_gates, specific_penalty

class FFN(nn.Module):
    def __init__(self, in_dims, emb_dims, b=256):
        super().__init__()
        self.w1 = nn.Linear(in_dims, b, bias=False)
        self.act1 = nn.LeakyReLU()
        self.w3 = nn.Linear(b, b, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.table = nn.Linear(b, emb_dims, bias=False)
        self.a = nn.Parameter(torch.zeros(1, 1))

    def forward(self, x):
        b, l, d = x.shape
        v = x.view(-1, d)
        v = self.act1(self.w1(v))
        v = self.w3(v) + v * self.a
        v = self.softmax(v)
        v = self.table(v)
        return v.view(b, l, -1)


class ValueEncoder(nn.Module):
    def __init__(self, emb_dims):
        super().__init__()
        self.value_enc = FFN(1, emb_dims)
        self.mask_emb = nn.Parameter(torch.zeros(1, 1, emb_dims))

    def forward(self, x):
        if x.dim() == 3:
            unmask, expr = torch.chunk(x, 2, dim=-1)
            unmasked = self.value_enc(expr) * unmask
            masked = self.mask_emb * (1 - unmask)
            expr_emb = masked + unmasked
        else:
            expr = x.unsqueeze(-1)
            unmask = torch.ones_like(expr)
            expr_emb = self.value_enc(expr)
        return expr_emb, unmask

class ValueDecoder_00(nn.Module):
    def __init__(self, emb_dims, dropout=0.1, zero=False):
        super().__init__()
        self.zero = zero

        self.mlp = nn.Sequential(
            nn.LayerNorm(emb_dims),
            nn.Linear(emb_dims, emb_dims),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dims, 1)
        )

        if self.zero:
            self.zero_logit = nn.Sequential(
                nn.LayerNorm(emb_dims),
                nn.Linear(emb_dims, emb_dims),
                nn.LeakyReLU(),
                nn.Linear(emb_dims, emb_dims),
                nn.LeakyReLU(),
                nn.Linear(emb_dims, 1),
                nn.Sigmoid()
            )

    def forward(self, expr_emb):
        b, l, d = expr_emb.shape
        pred = self.mlp(expr_emb).view(b, l)
        if not self.zero:
            return pred
        zero_prob = self.zero_logit(expr_emb).view(b, l)
        return pred, zero_prob


class ValueDecoder_1(nn.Module):
    def __init__(self, emb_dims, dropout, zero=False):
        super().__init__()
        self.zero = zero

        self.norm = nn.LayerNorm(emb_dims)  # 加入 LayerNorm
        self.w1 = nn.Linear(emb_dims, emb_dims, bias=True)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(emb_dims, 1, bias=True)
        init.xavier_normal_(self.w1.weight, gain=1.0)
        init.xavier_normal_(self.w2.weight, gain=1.0)
        if self.zero:
            self.zero_logit = nn.Sequential(
                nn.Linear(emb_dims, emb_dims),
                nn.LeakyReLU(),
                nn.Linear(emb_dims, emb_dims),
                nn.LeakyReLU(),
                nn.Linear(emb_dims, 1),
                nn.Sigmoid(),
            )

    def forward(self, expr_emb):
        b, l, d = expr_emb.shape
        # pred = self.w2(self.act(self.w1(expr_emb))).view(b, l)
        x = self.norm(expr_emb)               # LayerNorm 
        x = self.w1(x)                        # Linear
        x = self.act(x)                       # ReLU 
        x = self.dropout(x)
        pred = self.w2(x).view(b, l)          # output

        if not self.zero:
            return pred
        zero_prob = self.zero_logit(expr_emb).view(b, l)
        return pred, zero_prob

class CellwiseDecoder_00(nn.Module):
    def __init__(self, in_dims, emb_dims=None, dropout=0.1, zero=False):
        super().__init__()
        emb_dims = emb_dims or in_dims
        self.zero = zero

        self.query_proj = nn.Sequential(
            nn.LayerNorm(in_dims),
            nn.Linear(in_dims, emb_dims),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dims, emb_dims),
        )
        self.key_proj = nn.Linear(in_dims, emb_dims)

        if self.zero:
            self.zero_logit = nn.Sequential(
                nn.LayerNorm(emb_dims),
                nn.Linear(emb_dims, emb_dims),
                nn.ReLU(),
                nn.Linear(emb_dims, emb_dims),
                nn.Sigmoid()
            )

    def forward(self, cell_emb, gene_emb):
        b, l, d = gene_emb.shape
        # Project gene embeddings into query space
        query = self.query_proj(gene_emb)  # [B, L, D]
        # Project cell embedding into key space
        key = self.key_proj(cell_emb).view(b, -1, 1)  # [B, D, 1]
        # Inner product as similarity
        pred = torch.bmm(query, key).view(b, l)
        if not self.zero:
            return pred
        zero_prob = self.zero_logit(query).bmm(key).view(b, l)
        return pred, zero_prob


class CellwiseDecoder_1(nn.Module):
    def __init__(self, in_dims, emb_dims=None, dropout=0.0, zero=False):
        super().__init__()
        emb_dims = emb_dims or in_dims
        self.zero = zero

        self.norm = nn.LayerNorm(in_dims)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.map = nn.Linear(in_dims, emb_dims)
        init.xavier_normal_(self.map.weight, gain=1.0)
        if zero:
            self.zero_logit = nn.Linear(emb_dims, emb_dims)

    def forward(self, cell_emb, gene_emb):
        b = cell_emb.size(0)
        # query = torch.sigmoid(self.map(gene_emb))
        # query = torch.sigmoid(self.map(self.act(self.norm(gene_emb))))
        # LayerNorm → ReLU → Dropout → Linear → Sigmoid
        x = self.norm(gene_emb)
        x = self.act(x)
        x = self.dropout(x)
        query = torch.sigmoid(self.map(x))  # shape: [b, l, d]
        key = cell_emb.view(b, -1, 1)
        pred = torch.bmm(query, key).view(b, -1)
        if not self.zero:
            return pred
        zero_query = self.zero_logit(gene_emb)
        zero_prob = torch.sigmoid(torch.bmm(zero_query, key)).view(b, -1)
        return pred, zero_prob


class ValueDecoder(nn.Module):
    def __init__(self, emb_dims, dropout, zero=False):
        super().__init__()
        self.zero = zero
        self.w1 = nn.Linear(emb_dims, emb_dims, bias=False)
        self.act = nn.LeakyReLU()
        self.w2 = nn.Linear(emb_dims, 1, bias=False)
        if self.zero:
            self.zero_logit = nn.Sequential(
                nn.Linear(emb_dims, emb_dims),
                nn.LeakyReLU(),
                nn.Linear(emb_dims, emb_dims),
                nn.LeakyReLU(),
                nn.Linear(emb_dims, 1),
                nn.Sigmoid(),
            )

    def forward(self, expr_emb):
        b, l, d = expr_emb.shape
        pred = self.w2(self.act(self.w1(expr_emb))).view(b, l)
        if not self.zero:
            return pred
        zero_prob = self.zero_logit(expr_emb).view(b, l)
        return pred, zero_prob


class CellwiseDecoder(nn.Module):
    def __init__(self, in_dims, emb_dims=None, dropout=0.0, zero=False):
        super().__init__()
        emb_dims = emb_dims or in_dims
        self.zero = zero
        self.map = nn.Linear(in_dims, emb_dims, bias=False)
        if zero:
            self.zero_logit = nn.Linear(emb_dims, emb_dims)

    def forward(self, cell_emb, gene_emb):
        b = cell_emb.size(0)
        query = torch.sigmoid(self.map(gene_emb))
        key = cell_emb.view(b, -1, 1)
        pred = torch.bmm(query, key).squeeze(-1)
        if not self.zero:
            return pred
        zero_query = self.zero_logit(gene_emb)
        zero_prob = torch.sigmoid(torch.bmm(zero_query, key)).squeeze(-1)
        return pred, zero_prob
