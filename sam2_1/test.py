import os
import argparse
from tqdm import tqdm

import cv2
import torch
import numpy as np
from models.ooal import Net as model

from utils.viz import viz_pred_test
from utils.util import set_seed, process_gt, normalize_map
from utils.evaluation import cal_kl, cal_sim, cal_nss

parser = argparse.ArgumentParser()

# 新加入的对齐 baseline 的“评测后处理”
parser.add_argument('--sigma_eval', type=float, default=0.0,
                    help='Gaussian blur sigma used for evaluation preprocessing (0 = off)')
##  path
parser.add_argument('--data_root', type=str, default='./dataset/')
parser.add_argument('--model_file', type=str, default=None)
parser.add_argument('--save_path', type=str, default='./save_preds')
##  image
parser.add_argument('--divide', type=str, default='Seen')
parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--resize_size', type=int, default=256)
#### test
parser.add_argument("--test_batch_size", type=int, default=1)
parser.add_argument('--test_num_workers', type=int, default=8)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--viz', action='store_true', default=False)

args = parser.parse_args()
args.mask_root = os.path.join(args.data_root, args.divide, "testset", "GT")

if args.viz:
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

if __name__ == '__main__':
    set_seed(seed=321)

    from data.agd20k_ego import TestData, SEEN_AFF, UNSEEN_AFF

    args.class_names = SEEN_AFF if args.divide == 'Seen' else UNSEEN_AFF

    testset = TestData(data_root=args.data_root, divide=args.divide, crop_size=args.crop_size)
    TestLoader = torch.utils.data.DataLoader(dataset=testset,
                                             batch_size=args.test_batch_size,
                                             shuffle=False,
                                             num_workers=args.test_num_workers,
                                             pin_memory=True)

    model = model(args, 768, 512).cuda()

    KLs, SIM, NSS = [], [], []
    model.eval()
    with torch.no_grad():
        for step, (image, gt_aff, obj, mask_path) in enumerate(tqdm(TestLoader)):
            # 【需要改1】确保 dtype & device
            image = image.cuda(non_blocking=True)  # [B, 3, H, W]
            gt_aff = gt_aff.long().cuda(non_blocking=True)  # [B]，必须 long 才能做通道索引

            # # 【需要改2】直接让模型返回选中类别的概率图 [B, H, W]
            # ego_pred = model(image, gt_aff=gt_aff)  # <- 你的模型 test 分支返回的就是 prob
            # # 若 test_batch_size=1，可以 squeeze；如果批量评测就不要 squeeze
            # if ego_pred.size(0) == 1:
            #     ego_pred = ego_pred.squeeze(0)  # [H, W]
            # # 转到CPU做后处理/评估
            # ego_pred = ego_pred.detach().cpu().numpy()
            # # 下面保持你原来的评估/可视化流程
            # ego_pred = normalize_map(ego_pred, args.crop_size)
            # # 读 GT，resize 到 crop_size，再计算指标
            # key = obj[0] if isinstance(obj, (list, tuple)) else obj
            # mp = mask_path[0] if isinstance(mask_path, (list, tuple)) else mask_path
            # GT_mask = process_gt(mp, args.crop_size)  # 你已有的工具函数（0/1）
            # kld = cal_kl(ego_pred, GT_mask)
            # sim = cal_sim(ego_pred, GT_mask)
            # nss = cal_nss(ego_pred, GT_mask)

            # === 1) 前向：按 gt_aff 选通道，返回 [B,H,W] 概率图 ===
            ego_pred = model(image, gt_aff=gt_aff)  # 已是 prob（不是 logits）

            # 若 batch_size==1，可 squeeze 成 [H,W]（批评测就不要 squeeze）
            if ego_pred.size(0) == 1:
                ego_pred = ego_pred.squeeze(0)  # [H, W]

            pm = ego_pred.detach().cpu().numpy().astype(np.float32)

            # === 2) 评测前处理：min-max 到 [0,1]；可选 Gaussian blur；KLD 用 L1 归一化 ===
            pm = (pm - pm.min()) / (pm.max() - pm.min() + 1e-8)  # 给 SIM/NSS 的 [0,1] map
            if args.sigma_eval > 0:
                pm = cv2.GaussianBlur(pm, (0, 0), args.sigma_eval)

            p = pm / (pm.sum() + 1e-12)  # 给 KLD 的概率分布

            # === 3) 你原来的 key / mask 解析逻辑（保留） ===
            key = obj[0] if isinstance(obj, (list, tuple)) else obj
            mp = mask_path[0] if isinstance(mask_path, (list, tuple)) else mask_path

            # 读取并对齐 GT（你已有的工具函数，输出 0/1 或 [0,1]）
            GT_mask = process_gt(mp, args.crop_size).astype(np.float32)

            # === 4) 给指标分别准备 GT 版本 ===
            GT01 = (GT_mask - GT_mask.min()) / (GT_mask.max() - GT_mask.min() + 1e-8)  # 给 SIM/NSS
            q = GT01 / (GT01.sum() + 1e-12)  # 给 KLD 的概率分布

            # === 5) 指标：KLD 用 p vs q；SIM/NSS 用 pm vs GT01 ===
            kld = cal_kl(p, q)
            sim = cal_sim(pm, GT01)
            nss = cal_nss(pm, GT01)

            KLs.append(kld);
            SIM.append(sim);
            NSS.append(nss)

            if args.viz:
                img_name = key.split(".")[0]
                viz_pred_test(args, image, ego_pred, GT_mask, args.class_names, gt_aff, img_name)

    mKLD = sum(KLs) / len(KLs) if KLs else 0.0
    mSIM = sum(SIM) / len(SIM) if SIM else 0.0
    mNSS = sum(NSS) / len(NSS) if NSS else 0.0
    print(f"mKLD = {mKLD:.3f} mSIM = {mSIM:.3f} mNSS = {mNSS:.3f}")
    # assert os.path.exists(args.model_file), "Please provide the correct model file for testing"
    # state_dict = torch.load(args.model_file)['model_state_dict']
    # model.load_state_dict(state_dict, strict=False)
    #
    # GT_path = args.divide + "_gt.t7"
    # if not os.path.exists(GT_path):
    #     process_gt(args)
    # GT_masks = torch.load(GT_path)
    #
    # for step, (image, gt_aff, object, mask_path) in enumerate(tqdm(TestLoader)):
    #     ego_pred = model(image.cuda(), gt_aff=gt_aff)
    #     ego_pred = np.array(ego_pred.squeeze().data.cpu())
    #     ego_pred = normalize_map(ego_pred, args.crop_size)
    #
    #     names = mask_path[0].split("/")
    #     key = names[-3] + "_" + names[-2] + "_" + names[-1]
    #
    #     if key in GT_masks.keys():
    #         GT_mask = GT_masks[key]
    #         GT_mask = GT_mask / 255.0
    #
    #         GT_mask = cv2.resize(GT_mask, (args.crop_size, args.crop_size))
    #
    #         kld, sim, nss = cal_kl(ego_pred, GT_mask), cal_sim(ego_pred, GT_mask), cal_nss(ego_pred, GT_mask)
    #
    #         KLs.append(kld)
    #         SIM.append(sim)
    #         NSS.append(nss)
    #
    #     if args.viz:
    #         img_name = key.split(".")[0]
    #         viz_pred_test(args, image, ego_pred, GT_mask, args.class_names, gt_aff, img_name)
    #
    # mKLD, mSIM, mNSS = sum(KLs) / len(KLs), sum(SIM) / len(SIM), sum(NSS) / len(NSS)
    #
    # print(
    #     "mKLD = " + str(round(mKLD, 3)) + " mSIM = " + str(round(mSIM, 3)) + " mNSS = " + str(round(mNSS, 3))
    # )
