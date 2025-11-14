import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
from models.qformer import QueryFormer
from models.seg_decoder import tokens_to_grid, upsample_to_image

from models.clip import clip
from models.coop import TextEncoder, PromptLearner
from models.seg_decoder import SegDecoder
from models.seg_decoder import tokens_to_grid, upsample_to_image  # 顶部导入


def load_clip_to_cpu(backbone_name):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Net(nn.Module):
    def __init__(self, args, input_dim, out_dim,
                 dino_repo_dir,
                 dino_weights_path,
                 dino_pretrained='dinov3_vitb16'):
        super().__init__()
        self.dino_pretrained = dino_pretrained
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.class_names = args.class_names
        self.num_aff = len(self.class_names)

        # set up a vision embedder
        self.embedder = Mlp(in_features=input_dim, hidden_features=int(out_dim), out_features=out_dim,
                            act_layer=nn.GELU, drop=0.)
        # self.dino_model = torch.hub.load('facebookresearch/dinov2', self.dino_pretrained).cuda()
        # === 只保留 DINOv3 的本地 Hub 加载 ===
        assert dino_repo_dir is not None and dino_weights_path is not None, \
            "Please set dino_repo_dir (local cloned dinov3 repo) and dino_weights_path (*.pth)."
        assert os.path.exists(dino_repo_dir), f"dinov3 repo dir not found: {dino_repo_dir}"
        assert os.path.isfile(dino_weights_path), f"dinov3 weights .pth not found: {dino_weights_path}"
        self.dino_model = torch.hub.load(
            dino_repo_dir,  # 例如 r"C:\Users\H7z\Desktop\J\FYP\vision\dinov3"
            self.dino_pretrained,  # 'dinov3_vitb16'
            source='local',
            weights=dino_weights_path  # 你的 .pth
        ).cuda().eval()

        clip_model = load_clip_to_cpu('ViT-B/16').float()
        classnames = [a.replace('_', ' ')for a in self.class_names]
        self.prompt_learner = PromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.aff_text_encoder = TextEncoder(clip_model)

        # self.seg_decoder = SegDecoder(embed_dims=out_dim, num_layers=2)
        self.seg_decoder = QueryFormer(
            embed_dims=out_dim,
            num_layers=3,  # 你也可以先 2 再 3
            num_heads=8,
            num_queries=32,  # 16/32/64 试验，32 常见
            dropout=0.1,
            attn_temp=2,  # τ_attn
            cls_temp=2.5  # τ_cls
        )

        self.merge_weight = nn.Parameter(torch.zeros(3))

        self.lln_linear = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(3)])
        self.lln_norm = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(3)])

        self.lln_norm_1 = nn.LayerNorm(out_dim)
        self.lln_norm_2 = nn.LayerNorm(out_dim)

        self.linear_cls = nn.Linear(input_dim, out_dim)

        self._freeze_stages(exclude_key=['embedder', 'ctx', 'seg_decoder', 'lln_', 'merge_weight', 'linear_cls'])

        # === 新增：解冻 dinov3 的最后 1 层（或 2 层） ===
        # 自动找出 dinov3 backbone 的最大 blocks 索引，再解冻它（们）
        max_block_idx = -1
        for n, _ in self.dino_model.named_parameters():
            # 形如 "...blocks.11...."
            if "blocks." in n:
                try:
                    idx = int(n.split("blocks.")[1].split(".")[0])
                    if idx > max_block_idx:
                        max_block_idx = idx
                except Exception:
                    pass

        # 要解冻的 blocks 索引（最后 1 层；想解冻两层就用 [max_block_idx, max_block_idx-1] 前提 idx>=1）
        unfreeze_blocks = [max_block_idx] if max_block_idx >= 0 else []

        for name, param in self.dino_model.named_parameters():
            # 解冻最高层的 blocks，及其后面的 LayerNorm
            if any(f"blocks.{i}." in name for i in unfreeze_blocks) or (".norm" in name):
                param.requires_grad = True


    # 逐步解冻 DINOv3 顶部若干 block（以及最终的 norm）
    def progressive_unfreeze(self, extra_blocks: int = 2):  # 解冻了最后两层【norm+倒数二三层】
        """
        extra_blocks: 本次要额外解冻的“最顶层”block 个数（1~2 比较稳）
        """
        # 1) 找出当前 backbone 最大的 block 索引
        max_idx = -1
        for n, _ in self.dino_model.named_parameters():
            if "blocks." in n:
                try:
                    idx = int(n.split("blocks.")[1].split(".")[0])
                    max_idx = max(max_idx, idx)
                except Exception:
                    pass

        # 2) 计算需要解冻的目标 block（从顶部往下数）
        target = set()
        for k in range(extra_blocks):
            i = max_idx - k
            if i >= 0:
                target.add(i)

        # 3) 打开这些层以及最终 norm 的 requires_grad
        for name, p in self.dino_model.named_parameters():
            if any(f"blocks.{i}." in name for i in target) or (".norm" in name):
                p.requires_grad = True

    def forward(self, img, label=None, gt_aff=None):
        # {
        #     "x_norm_clstoken": x_norm[:, 0],
        #     "x_norm_patchtokens": x_norm[:, 1:],
        #     "x_prenorm": x,
        #     "masks": masks,
        # }

        b, _, h, w = img.shape

        # Last N features from DINO —— 兼容是否返回 CLS
        try:
            dino_out = self.dino_model.get_intermediate_layers(img, n=3, return_class_token=True)
            patches_list = [x[0] for x in dino_out]  # 每个 [B, L, C]
            cls_token = dino_out[-1][1]  # [B, C]
        except TypeError:
            # 某些版本不支持 return_class_token；只返回 patch tokens
            dino_out = self.dino_model.get_intermediate_layers(img, n=3)
            patches_list = dino_out  # 每个 [B, L, C]
            cls_token = patches_list[-1].mean(dim=1)  # 用平均当“伪 CLS”

        merge_weight = torch.softmax(self.merge_weight, dim=0)

        dino_dense = 0
        for i, feat in enumerate(patches_list):
            feat_ = self.lln_linear[i](feat)
            feat_ = self.lln_norm[i](feat_)
            dino_dense += feat_ * merge_weight[i]

        dino_dense = self.lln_norm_1(self.embedder(dino_dense))  # [B, L, out_dim]

        dino_cls = self.linear_cls(cls_token)  # [B, out_dim]
        # Affordance Text Encoder（恢复原逻辑）
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.lln_norm_2(self.aff_text_encoder(prompts, tokenized_prompts))
        # Last N features from DINO 这里是论文中加权融合
        # dino_out = self.dino_model.get_intermediate_layers(img, n=3, return_class_token=True)
        # merge_weight = torch.softmax(self.merge_weight, dim=0)
        #
        # dino_dense = 0
        # for i, feat in enumerate(dino_out):
        #     feat_ = self.lln_linear[i](feat[0])
        #     feat_ = self.lln_norm[i](feat_)
        #     dino_dense += feat_ * merge_weight[i]
        #
        # dino_dense = self.lln_norm_1(self.embedder(dino_dense))
        #
        # # Affordance Text Encoder
        # prompts = self.prompt_learner()
        # tokenized_prompts = self.tokenized_prompts
        # text_features = self.lln_norm_2(self.aff_text_encoder(prompts, tokenized_prompts))
        #
        # dino_cls = dino_out[-1][1]      # b x 768
        # dino_cls = self.linear_cls(dino_cls)

        text_features = text_features.unsqueeze(0).expand(b, -1, -1)
        text_features, attn_out, _ = self.seg_decoder(text_features, dino_dense, extra_token=dino_cls)

        # 计算 cross-attn（注意缩放用 out_dim 而不是固定 512）
        scale = (self.out_dim ** -0.5)
        attn = (text_features[-1] @ dino_dense.transpose(-2, -1)) * scale  # [B, N_aff, L]
        attn_out = torch.sigmoid(attn)  # [B, N_aff, L]

        # === 用 DINO 的 patch size 精确还原网格 ===
        def _get_patch_size(m):
            if hasattr(m, "patch_size"):
                return m.patch_size if isinstance(m.patch_size, int) else m.patch_size[0]
            if hasattr(m, "patch_embed") and hasattr(m.patch_embed, "patch_size"):
                ps = m.patch_embed.patch_size
                return ps if isinstance(ps, int) else ps[0]
            return 16  # 兜底：DINOv3 ViT-B/16

        p = _get_patch_size(self.dino_model)  # DINOv3 应该返回 16
        attn_maps = tokens_to_grid(attn_out, img_hw=img.shape[-2:], patch=p)  # [B, C, H_p, W_p]
        pred = upsample_to_image(attn_maps, out_hw=img.shape[-2:])  # [B, C, H, W]


        # if self.training:
        #     assert not label == None, 'Label should be provided during training'
        #     loss_bce = nn.BCELoss()(pred, label / 255.0)
        #     loss_dict = {'bce': loss_bce}
        #     return pred, loss_dict
        if self.training:
            y = (label / 255.0).clamp(0, 1)
            loss_bce = F.binary_cross_entropy(pred, y)

            B, C, H, W = pred.shape
            pred_dist = torch.softmax(pred.reshape(B, C, -1), dim=-1)
            gt = y.reshape(B, C, -1)
            gt = gt / (gt.sum(-1, keepdim=True) + 1e-6)
            loss_kld = F.kl_div((pred_dist + 1e-8).log(), gt, reduction='mean')

            lam = 0.1  # 先小一点
            if getattr(self, "global_iter", 0) < 3000:
                loss = loss_bce
            else:
                loss = loss_bce + lam * loss_kld

            return pred, {"bce": loss_bce, "kld": loss_kld, "loss": loss}
        else:
            # ====== 新增：当 eval 且不指定 gt_aff 时  ======
            if gt_aff is None:  ### 为 [demo.py] 测试添加if，直接返回所有通道
                return pred  # [B, C(=36), H, W]
            # ====== 原来的逻辑保留，用于只取指定类 ======
            if gt_aff is not None:
                out = torch.zeros(b, h, w).cuda()
                for b_ in range(b):
                    out[b_] = pred[b_, gt_aff[b_]]
                return out

    def _freeze_stages(self, exclude_key=None):
        """Freeze stages param and norm stats."""
        for n, m in self.named_parameters():
            if exclude_key:
                if isinstance(exclude_key, str):
                    if not exclude_key in n:
                        m.requires_grad = False
                elif isinstance(exclude_key, list):
                    count = 0
                    for i in range(len(exclude_key)):
                        i_layer = str(exclude_key[i])
                        if i_layer in n:
                            count += 1
                    if count == 0:
                        m.requires_grad = False
                    elif count > 0:
                        print('Finetune layer in backbone:', n)
                else:
                    assert AttributeError("Dont support the type of exclude_key!")
            else:
                m.requires_grad = False
