import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# 添加sam2目录到Python路径
sam2_path = os.path.join(os.path.dirname(__file__), '..', 'sam2')
if sam2_path not in sys.path:
    sys.path.insert(0, sam2_path)

# 导入SAM2模块
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from models.clip import clip
from models.coop import TextEncoder, PromptLearner
from models.seg_decoder import SegDecoder


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


# def load_sam2_encoder(model_type = "sam2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml",
#                       checkpoint_path = "sam2/checkpoints/sam2.1_hiera_base_plus.pt"):
#     # 使用 SAM-2 的构建函数加载整个模型
#     sam2_model = build_sam2(model_type=model_type, checkpoint=checkpoint_path, device="cuda")
#     predictor = SAM2ImagePredictor(sam2_model)
#     return predictor

class Net(nn.Module):
    def __init__(self, args, input_dim, out_dim,
                 model_type="configs/sam2.1/sam2.1_hiera_b+.yaml",
                 checkpoint_path="sam2/checkpoints/sam2.1_hiera_base_plus.pt"
                 # dino_pretrained='vit_base_patch16'
                 ):
        super().__init__()
        # self.dino_pretrained = dino_pretrained
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.class_names = args.class_names
        self.num_aff = len(self.class_names)

        # set up a vision embedder
        self.embedder = Mlp(in_features=input_dim, hidden_features=int(out_dim), out_features=out_dim,
                            act_layer=nn.GELU, drop=0.)

        # self.dino_model = torch.hub.load('facebookresearch/mae', self.dino_pretrained).cuda()
        
        self.sam2_model = build_sam2(model_type, checkpoint_path, device="cuda")
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

        clip_model = load_clip_to_cpu('ViT-B/16').float()
        classnames = [a.replace('_', ' ')for a in self.class_names]
        self.prompt_learner = PromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.aff_text_encoder = TextEncoder(clip_model)

        self.seg_decoder = SegDecoder(embed_dims=out_dim, num_layers=2)

        self.merge_weight = nn.Parameter(torch.zeros(3))

        self.lln_linear = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(3)])
        self.lln_norm = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(3)])

        self.lln_norm_1 = nn.LayerNorm(out_dim)
        self.lln_norm_2 = nn.LayerNorm(out_dim)

        self.linear_cls = nn.Linear(input_dim, out_dim)

        # self._freeze_stages(exclude_key=['embedder', 'ctx', 'seg_decoder', 'lln_', 'merge_weight', 'linear_cls'])

        # Phase 1：只训练自家头部，SAM2 主干全部冻结
        self._freeze_for_finetune(phase='head')


    def forward(self, img, label=None, gt_aff=None):
        # {
        #     "x_norm_clstoken": x_norm[:, 0],
        #     "x_norm_patchtokens": x_norm[:, 1:],
        #     "x_prenorm": x,
        #     "masks": masks,
        # }

        b, _, h, w = img.shape
        # === SAM2 输入归一化 ===
        # 先把 0~255 转 0~1
        if img.dtype.is_floating_point:
            if img.max() > 1.5:
                img = img / 255.0
        else:
            img = img.float() / 255.0

        # 用 SAM2 预训练时的均值方差；若无属性就用常见 Detectron2/SAM 默认
        if hasattr(self.sam2_model, 'pixel_mean') and hasattr(self.sam2_model, 'pixel_std'):
            mean = (self.sam2_model.pixel_mean / 255.0).view(1, 3, 1, 1).to(img)
            std = (self.sam2_model.pixel_std / 255.0).view(1, 3, 1, 1).to(img)
        else:
            mean = img.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = img.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        img = (img - mean) / std
        # === 归一化结束 ===

        # 使用SAM2的forward_image方法获取中间特征
        backbone_out = self.sam2_model.forward_image(img)
        features = backbone_out["backbone_fpn"]
        
        # 模拟DINO的get_intermediate_layers返回格式
        # 返回格式: [[patch_tokens, class_token], ...] for each layer
        dino_out = []
        spatial_hw = []  # 新增：记录每层空间尺寸
        for i, feat in enumerate(features[-3:]):  # 取最后3层特征
            b, c, h, w = feat.shape
            # 将特征转换为类似DINO的格式 [patch_tokens, class_token]
            patch_tokens = feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
            # 创建虚拟的class token (使用全局平均池化)
            class_token = feat.mean(dim=[2, 3], keepdim=True).flatten(1)  # [B, C]
            dino_out.append([patch_tokens, class_token])
            spatial_hw.append((h, w))  # 新增
        
        # 调整特征维度以匹配原始DINO的768维
        for i, (patch_tokens, class_token) in enumerate(dino_out):
            if patch_tokens.shape[-1] != 768:
                # 添加线性层来调整维度
                if not hasattr(self, f'dim_adapter_{i}'):
                    setattr(self, f'dim_adapter_{i}', nn.Linear(patch_tokens.shape[-1], 768).cuda())
                patch_tokens = getattr(self, f'dim_adapter_{i}')(patch_tokens)
                class_token = getattr(self, f'dim_adapter_{i}')(class_token)
                dino_out[i] = [patch_tokens, class_token]
        
        
        
        
        merge_weight = torch.softmax(self.merge_weight, dim=0)

        dino_dense = 0
        H0, W0 = spatial_hw[0]
        for i, feat in enumerate(dino_out):
            # feat_ = self.lln_linear[i](feat[0])
            # feat_ = self.lln_norm[i](feat_)
            # # 调整空间分辨率以匹配第一个特征图
            # if i > 0:
            #     target_size = dino_out[0][0].shape[1]  # 使用第一个特征图的空间大小
            #     if feat_.shape[1] != target_size:
            #         # 使用插值调整空间大小
            #         feat_ = feat_.transpose(1, 2)  # [B, C, H*W]
            #         feat_ = feat_.view(feat_.shape[0], feat_.shape[1], int(feat_.shape[2]**0.5), int(feat_.shape[2]**0.5))
            #         feat_ = F.interpolate(feat_, size=(int(target_size**0.5), int(target_size**0.5)), mode='bilinear', align_corners=False)
            #         feat_ = feat_.flatten(2).transpose(1, 2)  # [B, H*W, C]
            #
            # dino_dense += feat_ * merge_weight[i]
            tokens = self.lln_linear[i](feat[0])  # [B, Ni, C]
            tokens = self.lln_norm[i](tokens)

            Hi, Wi = spatial_hw[i]
            b, Ni, C_ = tokens.shape
            fmap = tokens.transpose(1, 2).reshape(b, C_, Hi, Wi)  # [B, C, Hi, Wi]

            if (Hi, Wi) != (H0, W0):
                fmap = F.interpolate(fmap, size=(H0, W0), mode='bilinear', align_corners=False)

            tokens = fmap.flatten(2).transpose(1, 2)  # [B, H0*W0, C]
            dino_dense = dino_dense + tokens * merge_weight[i]

        dino_dense = self.lln_norm_1(self.embedder(dino_dense))

        # Affordance Text Encoder
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.lln_norm_2(self.aff_text_encoder(prompts, tokenized_prompts))

        dino_cls = dino_out[-1][1]      # b x 768
        dino_cls = self.linear_cls(dino_cls)

        text_features = text_features.unsqueeze(0).expand(b, -1, -1)
        text_features, attn_out, _ = self.seg_decoder(text_features, dino_dense, extra_token=dino_cls)

        attn = (text_features[-1] @ dino_dense.transpose(-2, -1)) * (512 ** -0.5)
        # attn_out = torch.sigmoid(attn)
        # attn_out = attn_out.reshape(b, -1, h // 14, w // 14)
        attn_map = attn.reshape(b, self.num_aff, H0, W0)
        pred = F.interpolate(
            attn_map, size=img.shape[-2:], mode="bilinear", align_corners=False
        )

        # if self.training:
        #     assert not label == None, 'Label should be provided during training'
        #     loss_bce = nn.BCELoss()(pred, label / 255.0)
        #     loss_dict = {'bce': loss_bce}
        #     return pred, loss_dict
        #
        # else:
        #     if gt_aff is not None:
        #         out = torch.zeros(b, h, w).cuda()
        #         for b_ in range(b):
        #             out[b_] = pred[b_, gt_aff[b_]]
        #         return out
        if self.training:
            assert label is not None, 'Label should be provided during training'

            b = img.size(0)

            # --- 统一 float、归一化到 [0,1] ---
            target = label
            if target.dtype != torch.float32:
                target = target.float()
            if target.max() > 1:
                target = target / 255.0

            # --- 情况A：单通道掩码（或 [B,1,H,W]） -> 按 gt_aff 选通道 ---
            if target.dim() == 3 or target.shape[1] == 1:
                assert gt_aff is not None, "training with single-channel GT requires gt_aff"
                gt_aff = gt_aff.long().view(-1).to(pred.device)  # [B]
                logits_used = pred[torch.arange(b, device=pred.device), gt_aff, :, :].unsqueeze(1)
                if target.dim() == 3:
                    target = target.unsqueeze(1)
                loss_bce = torch.nn.BCEWithLogitsLoss()(logits_used, target)

            # --- 情况B：one-hot 多通道标签 [B,num_aff,H,W] ---
            else:
                loss_bce = torch.nn.BCEWithLogitsLoss()(pred, target)

            loss_dict = {'bce': loss_bce}
            return pred, loss_dict

        else:
            pred_prob = torch.sigmoid(pred)
            if gt_aff is not None:
                # 推理/可视化时再做 sigmoid 得到概率
                pred_prob = torch.sigmoid(pred)
                if gt_aff is not None:
                    # 保证 gt_aff 为 long，并展平成 [B]
                    gt_aff = gt_aff.long().view(-1).to(pred_prob.device)  # [B]

                    # 批量索引，得到 [B, H, W]
                    out = pred_prob[torch.arange(b, device=pred_prob.device), gt_aff, :, :]
                    return out  # [B, H, W]
                else:
                    return pred_prob  # [B, C, H, W]

    def _freeze_for_finetune(self, phase='head'):
        """
        phase:
          'head'  -> 只训练自家头部（prompt_learner/seg_decoder/...），SAM2 主干全冻结
          'high'  -> 在 head 的基础上，再解冻 SAM2 的 FPN + 最后一个 stage（高层）
          'all'   -> 全部可训练（不建议直接用）
        """
        # 先全部冻结
        for n, p in self.named_parameters():
            p.requires_grad = False

        # 你的“头部”——建议始终可训练
        head_keys = [
            'prompt_learner',  # 覆盖 ctx
            'seg_decoder',
            'embedder',
            'lln_',
            'merge_weight',
            'linear_cls',
            'dim_adapter_',  # 动态创建的维度适配层
        ]
        for n, p in self.named_parameters():
            if any(k in n for k in head_keys):
                p.requires_grad = True

        if phase in ('high', 'all'):
            # 根据实际命名调整这些前缀（print(self) 看一下）
            sam2_high = ['sam2_model.neck', 'sam2_model.fpn',
                         'sam2_model.image_encoder.stages.3', 'sam2_model.image_encoder.layer4']
            for n, p in self.named_parameters():
                if any(k in n for k in sam2_high):
                    p.requires_grad = True

        if phase == 'all':
            for n, p in self.named_parameters():
                p.requires_grad = True

        # 可选：打印确认
        print('---- Trainable parameters ----')
        for n, p in self.named_parameters():
            if p.requires_grad:
                print('[TRAIN]', n)

    # def _freeze_stages(self, exclude_key=None):
    #     """Freeze stages param and norm stats."""
    #     for n, m in self.named_parameters():
    #         if exclude_key:
    #             if isinstance(exclude_key, str):
    #                 if not exclude_key in n:
    #                     m.requires_grad = False
    #             elif isinstance(exclude_key, list):
    #                 count = 0
    #                 for i in range(len(exclude_key)):
    #                     i_layer = str(exclude_key[i])
    #                     if i_layer in n:
    #                         count += 1
    #                 if count == 0:
    #                     m.requires_grad = False
    #                 elif count > 0:
    #                     print('Finetune layer in backbone:', n)
    #             else:
    #                 assert AttributeError("Dont support the type of exclude_key!")
    #         else:
    #             m.requires_grad = False
