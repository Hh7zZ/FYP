import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# --------- 基础 Cross-Attn（支持温度 & CLS 引导掩码）---------
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0,
                 attn_temp=1.5, cls_temp=2.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_temp = attn_temp
        self.cls_temp = cls_temp

    def forward(self, x_q, x_kv, cls_token=None):
        B, Nq, C = x_q.shape
        Nk = x_kv.shape[1]

        q = self.q(x_q).reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0,2,1,3)   # B h Nq d
        k = self.k(x_kv).reshape(B, Nk, self.num_heads, C // self.num_heads).permute(0,2,1,3)   # B h Nk d
        v = self.v(x_kv).reshape(B, Nk, self.num_heads, C // self.num_heads).permute(0,2,1,3)

        # logits + 温度（让注意力更“尖锐”）
        attn_logits = (q @ k.transpose(-2, -1)) * (self.scale / self.attn_temp)                # B h Nq Nk

        # CLS 引导掩码（CTM）：sigmoid(cls·K^T/τ_cls) → [0,1]
        aff_mask = None
        if cls_token is not None:
            cls_tok = cls_token.reshape(B, 1, self.num_heads, C // self.num_heads).permute(0,2,1,3)  # B h 1 d
            aff_mask = (cls_tok @ k.transpose(-2, -1)) * (self.scale / self.cls_temp)                # B h 1 Nk
            aff_mask = torch.sigmoid(aff_mask)                                                       # B h 1 Nk
            # 聚合到 head 维度、再标准化
            aff_mask = aff_mask.mean(dim=1, keepdim=True)                                            # B 1 1 Nk
            # per-image min-max 到 [0,1]
            m, M = aff_mask.amin(dim=-1, keepdim=True), aff_mask.amax(dim=-1, keepdim=True)
            aff_mask = (aff_mask - m) / (M - m + 1e-6)

            attn = torch.softmax(attn_logits, dim=-1)
            attn = attn * aff_mask
        else:
            attn = torch.softmax(attn_logits, dim=-1)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1,2).reshape(B, Nq, C)                                           # B Nq C
        x = self.proj(x)
        x = self.proj_drop(x)
        # 返回：输出、原始 logits（供可视化）、aff_mask
        return x, attn_logits.sum(dim=1) / self.num_heads, aff_mask    # B Nq C, B Nq Nk, B 1 1 Nk


# --------- 单层 Q-Former：Resampler + Text Cross-Attn + FFN ----------
class QFormerLayer(nn.Module):
    def __init__(self, dim, num_heads=8, num_queries=32, dropout=0.1, attn_temp=1.5, cls_temp=2.0):
        super().__init__()
        self.num_queries = num_queries
        self.query_embed = nn.Parameter(torch.randn(num_queries, dim) * 0.02)  # 可学习 queries

        self.ca_resampler = CrossAttention(dim, num_heads, qkv_bias=True,
                                           attn_drop=dropout, proj_drop=dropout,
                                           attn_temp=attn_temp, cls_temp=cls_temp)
        self.norm_r1 = nn.LayerNorm(dim)
        self.ffn_r = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(dropout),
                                   nn.Linear(dim*4, dim), nn.Dropout(dropout))
        self.norm_r2 = nn.LayerNorm(dim)

        # 文本 tokens 与“已压缩的视觉摘要”再做一次跨注意（对齐）
        self.ca_text = CrossAttention(dim, num_heads, qkv_bias=True,
                                      attn_drop=dropout, proj_drop=dropout,
                                      attn_temp=attn_temp, cls_temp=cls_temp)
        self.norm_t1 = nn.LayerNorm(dim)
        self.ffn_t = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(dropout),
                                   nn.Linear(dim*4, dim), nn.Dropout(dropout))
        self.norm_t2 = nn.LayerNorm(dim)

    def forward(self, text_tokens, vis_tokens, cls_token=None):
        B, Nt, C = text_tokens.shape
        # 1) Resampler：learnable queries 读原始视觉 token
        q = self.query_embed.unsqueeze(0).expand(B, -1, -1)             # B Q C
        res, attn_logits_res, aff_mask = self.ca_resampler(q, vis_tokens, cls_token)  # B Q C
        res = self.norm_r1(q + res)
        res = self.norm_r2(res + self.ffn_r(res))                        # B Q C  (condensed visual)

        # 2) Text ↔ Resampled Visual：文本查询只看“摘要” → 更稳的对齐
        t, attn_logits_txt, _ = self.ca_text(text_tokens, res, cls_token=None)        # B Nt C
        t = self.norm_t1(text_tokens + t)
        t = self.norm_t2(t + self.ffn_t(t))                                           # B Nt C

        # 返回文本侧（对齐后的）tokens；以及第一阶段对原始像素的注意（可当可视化）
        return t, attn_logits_res, aff_mask

# --------- 堆叠若干层 ----------
class QueryFormer(nn.Module):
    def __init__(self, embed_dims=512, num_layers=3, num_heads=8,
                 num_queries=32, dropout=0.1, attn_temp=1.5, cls_temp=2.0):
        super().__init__()
        self.layers = nn.ModuleList([
            QFormerLayer(embed_dims, num_heads, num_queries, dropout, attn_temp, cls_temp)
            for _ in range(num_layers)
        ])
        # --- 新增：初始化 ---
        self.reset_parameters()

    # --- 新增：系统化初始化 ---
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # queries 初始更小，避免一开始注意力铺满
        for l in self.layers:
            nn.init.normal_(l.query_embed, mean=0.0, std=0.01)

    # --- 新增：训练时动态调温度（从“软”到“尖”） ---
    def set_temperature(self, attn_temp: float, cls_temp: float):
        for l in self.layers:
            l.ca_resampler.attn_temp = attn_temp
            l.ca_resampler.cls_temp  = cls_temp
            l.ca_text.attn_temp      = attn_temp
            l.ca_text.cls_temp       = cls_temp

    def forward(self, queries, feat, extra_token=None):
        """
        queries: [B, N_aff, C]  文本 tokens
        feat:    [B, L, C]      视觉 patch tokens
        extra_token: [B, C]     CLS 引导（可为 None）
        return: (outputs_per_layer, attn_logits_per_layer, masks_per_layer)
        """
        out_list, attn_list, mask_list = [], [], []
        x = queries
        for layer in self.layers:
            x, attn_logits, aff_mask = layer(x, feat, extra_token)  # x: [B, N_aff, C]
            out_list.append(x)
            attn_list.append(attn_logits)    # [B, N_aff, L] (logits)
            mask_list.append(aff_mask)       # [B, 1, 1, L]  or None
        return out_list, attn_list, mask_list