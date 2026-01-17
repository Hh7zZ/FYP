import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
import argparse

from models.ooal import Net as OOALNet  # 载入主模型
from data.agd20k_ego import SEEN_AFF  # 36 个 seen affordances 的列表

# ======================
# 参数
# ======================
parser = argparse.ArgumentParser()
parser.add_argument('--img', type=str, required=True, help='输入图片路径')
parser.add_argument('--model_file', type=str, default='./save_models/20251101_Unfreeze2Layer/best_model_20000_0.729_0.594_1.721.pth', help='训练好的模型权重路径')
parser.add_argument('--aff', type=str, default='hold', help='只展示这些词（逗号分隔）；模型仍按 Seen 36 词建模')
parser.add_argument('--save_dir', type=str, default='./demo_result/Unseen', help='保存结果路径')
parser.add_argument('--resize_size', type=int, default=256)
parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--alpha', type=float, default=0.6)
parser.add_argument('--dino_repo_dir', type=str, default='./dinov3-main/dinov3-main')
parser.add_argument('--dino_weights_path', type=str, default='./dinov3-main/dinov3-main/dinov3/checkpointer/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth')
parser.add_argument('--dino_pretrained', type=str, default='dinov3_vitb16')

parser.add_argument('--top_keep', type=float, default=10,  help='仅保留最强的 top-% 像素，收紧区域')
parser.add_argument('--gamma',    type=float, default=2.0,   help='热力图 gamma（>1 越收紧）')
parser.add_argument('--morph_iter', type=int, default=0,     help='形态学腐蚀迭代次数，0 关闭')

args = parser.parse_args()

# ======================
# Step 1. 读取图片
# ======================
os.makedirs(args.save_dir, exist_ok=True)
raw = Image.open(args.img).convert('RGB')

# 与训练一致的预处理
tfm = transforms.Compose([
    transforms.Resize((args.resize_size, args.resize_size)),
    transforms.CenterCrop(args.crop_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711]),
])
img = tfm(raw).unsqueeze(0).cuda()  # [1,3,H,W]

# ======================
# Step 2. 构建模型并加载权重
# ======================
class Args: pass
args_net = Args()
args_net.class_names = SEEN_AFF  # ★ 关键：始终用训练时的 36 个 Seen 词
model = OOALNet(args_net, 768, 512,
                dino_repo_dir=args.dino_repo_dir,
                dino_weights_path=args.dino_weights_path,
                dino_pretrained=args.dino_pretrained).cuda().eval()

print(f"Loading trained weights from: {args.model_file}")
state = torch.load(args.model_file, map_location='cpu')
if 'model_state_dict' in state:
    state = state['model_state_dict']
model.load_state_dict(state, strict=False)

# ======================
# Step 3. 前向推理（不提供gt_aff，返回所有通道）
# ======================
model.eval().cuda()
with torch.no_grad():
    pred = model(img, None)                 # 现在会返回 [1, 36, H, W]
    if isinstance(pred, tuple):
        pred = pred[0]
    # 如果你不确定 pred 是否已是 [0,1]，可以安全地再过一次 sigmoid（可选）
    # pred = torch.sigmoid(pred)
    pred = pred.squeeze(0).cpu().numpy()    # [36, H, W]

# ======================
# Step 4. 只对 --aff 指定的词叠加热图
# ======================
want = [w.strip().lower() for w in args.aff.split(',') if w.strip()]
name2idx = {n.lower(): i for i, n in enumerate(SEEN_AFF)}

# img_bgr = cv2.cvtColor(np.array(raw.resize((pred.shape[2], pred.shape[1]))), cv2.COLOR_RGB2BGR)
img_bgr = cv2.cvtColor(np.array(raw), cv2.COLOR_RGB2BGR)
h_orig, w_orig = img_bgr.shape[:2] # 获取原图高宽
for verb in want:
    if verb not in name2idx:
        print(f"[WARN] '{verb}': 不在 Seen 36 词里，跳过")
        continue

    i = name2idx[verb]
    m = pred[i]  # [H, W] 概率（0~1，可能很小）

    # # --- 可视化增强：分位数对比度拉伸（先把弱响应抬起来）
    # lo, hi = np.percentile(m, (50, 99.5))  # 需要时可把 50/99.5 调整
    # # m = (m - lo) / (hi - lo + 1e-6)
    # # m = np.clip(m, 0.0, 1.0)
    # # 如果想用最简单的 min-max 归一化，也可以用下面两行替换上面的分位数拉伸：
    # m = (m - m.min()) / (m.max() - m.min() + 1e-6)
    # m = np.clip(m, 0.0, 1.0)
    # heat = (m * 255).astype(np.uint8)
    # heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    # over = cv2.addWeighted(img_bgr, 1 - args.alpha, heat_color, args.alpha, 0)

    # ------- 自适应阈值与增强 -------
    pmax = float(m.max())  ###
    # 1) 分位数拉伸（放宽到 40/99.8，更容易“看到”弱响应）
    lo, hi = np.percentile(m, (40, 99.8))
    m = np.clip((m - lo) / (hi - lo + 1e-6), 0.0, 1.0)

    # 2) 如果通道整体很弱（例如 <0.15），兜底：不做 top-k、不做腐蚀，只做轻度 gamma
    if pmax < 0.15:
        GAMMA = 1.2
        m = m ** GAMMA
    else:
        # 通道还可以：适度保留 top 区域 + 温和 gamma
        TOP_KEEP = 25.0  # 先放宽到 25%，再根据效果降到 15~20
        thr = np.percentile(m, 100.0 - TOP_KEEP)
        m = np.clip((m - thr) / (1.0 - thr + 1e-6), 0.0, 1.0)

        GAMMA = 1.4  # 比之前 2.0 温和，避免剪太狠
        m = m ** GAMMA

        # 不建议腐蚀，刀刃很细会被抹掉；如必须，可把 ITER=1 再试
        ITER = 0
        if ITER > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            m_bin = ((m > 0).astype(np.uint8)) * 255
            m_bin = cv2.erode(m_bin, k, iterations=ITER)
            m = m * (m_bin > 0)

    # 3) 轻度平滑（让响应连贯，不影响中心位置）
    m = cv2.GaussianBlur(m.astype(np.float32), (3, 3), 0)

    # ------- 上色与叠加 -------
    m_resized = cv2.resize(m, (w_orig, h_orig))  # 拉伸 mask
    heat = (np.clip(m_resized, 0, 1) * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    over = cv2.addWeighted(img_bgr, 1 - args.alpha, heat_color, args.alpha, 0)
    # heat = (np.clip(m, 0, 1) * 255).astype(np.uint8)
    # heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    # over = cv2.addWeighted(img_bgr, 1 - args.alpha, heat_color, args.alpha, 0)

    out_path = os.path.join(args.save_dir, f"{os.path.splitext(os.path.basename(args.img))[0]}_{verb}.png")
    cv2.imwrite(out_path, over)
    print("Saved:", out_path)

print(f"\nAll done! Results saved to: {args.save_dir}")
