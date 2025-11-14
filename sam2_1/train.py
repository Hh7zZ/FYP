import os
import sys
import time
import shutil
import logging
import argparse


# 添加当前目录到Python路径，确保能导入utils等模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import cv2
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from models.ooal import Net as model

from utils.viz import viz_pred_test
from utils.util import set_seed, process_gt, normalize_map, get_optimizer
from utils.evaluation import cal_kl, cal_sim, cal_nss
from pathlib import Path

parser = argparse.ArgumentParser()
##  path
parser.add_argument('--data_root', type=str, default='./dataset/')
parser.add_argument('--save_root', type=str, default='save_models')
##  image
parser.add_argument('--divide', type=str, default='Seen')
parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--resize_size', type=int, default=256)
##  dataloader
parser.add_argument('--num_workers', type=int, default=0)
##  train
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--warm_epoch', type=int, default=0)
parser.add_argument('--iters', type=int, default=20000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--show_step', type=int, default=100)
parser.add_argument('--eval_step', type=int, default=2000)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--viz', action='store_true', default=False)

#### test
parser.add_argument("--test_batch_size", type=int, default=1)
parser.add_argument('--test_num_workers', type=int, default=0)

args = parser.parse_args()
torch.cuda.set_device('cuda:' + args.gpu)

time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
args.save_path = os.path.join(args.save_root, time_str)
args.mask_root = os.path.join(args.data_root, args.divide, "testset", "GT")

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path, exist_ok=True)
dict_args = vars(args)

str_1 = ""
for key, value in dict_args.items():
    str_1 += key + "=" + str(value) + "\n"

logging.basicConfig(filename='%s/run.log' % args.save_path, level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.info(str_1)

if __name__ == '__main__':
    # set_seed(seed=321)

    from data.agd20k_ego import TrainData, TestData, SEEN_AFF, UNSEEN_AFF

    args.class_names = SEEN_AFF if args.divide == 'Seen' else UNSEEN_AFF

    trainset = TrainData(data_root=args.data_root,
                         divide=args.divide,
                         resize_size=args.resize_size,
                         crop_size=args.crop_size)

    TrainLoader = torch.utils.data.DataLoader(dataset=trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers,
                                              pin_memory=True,
                                              drop_last=True)

    testset = TestData(data_root=args.data_root, divide=args.divide, crop_size=args.crop_size)
    TestLoader = torch.utils.data.DataLoader(dataset=testset,
                                             batch_size=args.test_batch_size,
                                             shuffle=False,
                                             num_workers=args.test_num_workers,
                                             pin_memory=True,
                                             drop_last=True)


    model = model(args, 768, 512).cuda()
    model.train()
    optimizer, scheduler = get_optimizer(model, args)

    best_kld = 1000
    total_iter = 0
    print('Train begining!')

    while True:
        for _, (img, ann) in enumerate(TrainLoader):
            img, ann = img.cuda(), ann.cuda().float()
            pred, loss_dict = model(img, label=ann)

            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            # 加一行：梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            if (total_iter + 1) % args.show_step == 0:
                log_str = 'iters: %d/%d | ' % (total_iter + 1, args.iters)
                log_str += ' | '.join(['%s: %.3f' % (k, v) for k, v in loss_dict.items()])
                log_str += ' | '
                log_str += 'lr {:.6f}'.format(scheduler.get_last_lr()[0])
                logger.info(log_str)

            total_iter += 1

            if (total_iter + 1) % args.eval_step == 0:

                KLs, SIM, NSS = [], [], []
                model.eval()
                GT_path = args.divide + "_gt.t7"
                if not os.path.exists(GT_path):
                    process_gt(args)
                GT_masks = torch.load(GT_path, weights_only=False)

                for step, (image, gt_aff, object, mask_path) in enumerate(tqdm(TestLoader)):
                    ego_pred = model(image.cuda(), gt_aff=gt_aff)
                    ego_pred = np.array(ego_pred.squeeze().data.cpu())
                    ego_pred = normalize_map(ego_pred, args.crop_size)
                    # 尺寸对齐 + 基线一致的 min-max 归一化到 [0,1]
                    emin, emax = ego_pred.min(), ego_pred.max()
                    ego_pred = (ego_pred - emin) / (emax - emin + 1e-8)  # [0,1]

                    mp = Path(mask_path[0])
                    key = f"{mp.parent.parent.name}_{mp.parent.name}_{mp.name}"

                    GT_mask = GT_masks[key]
                    GT_mask = GT_mask / 255.0
                    GT_mask = cv2.resize(GT_mask, (args.crop_size, args.crop_size))
                    # 与基线一致：对 GT 也做 min-max 归一化到 [0,1]
                    gmin, gmax = GT_mask.min(), GT_mask.max()
                    GT_mask = (GT_mask - gmin) / (gmax - gmin + 1e-8)
                    # ====== 计算指标 ======
                    # KLD 需要“分布”（和为 1）；SIM/NSS 用 [0,1] 就行
                    p = ego_pred / (ego_pred.sum() + 1e-12)
                    q = GT_mask / (GT_mask.sum() + 1e-12)
                    kld = cal_kl(p, q)
                    sim = cal_sim(ego_pred, GT_mask)
                    nss = cal_nss(ego_pred, GT_mask)
                    # kld, sim, nss = cal_kl(ego_pred, GT_mask), cal_sim(ego_pred, GT_mask), cal_nss(ego_pred, GT_mask)

                    KLs.append(kld)
                    SIM.append(sim)
                    NSS.append(nss)

                    # Visualization the prediction during evaluation
                    if args.viz:
                        if (step + 1) % 40 == 0:
                            img_name = key.split(".")[0]
                            viz_pred_test(args, image, ego_pred, GT_mask, args.class_names, gt_aff, img_name, total_iter)

                mKLD, mSIM, mNSS = sum(KLs) / len(KLs), sum(SIM) / len(SIM), sum(NSS) / len(NSS)

                logger.info(
                    "iter=" + str(total_iter + 1) + ' | ' + args.divide + ": mKLD = " + str(round(mKLD, 3))
                    + " mSIM = " + str(round(mSIM, 3)) + " mNSS = " + str(round(mNSS, 3)) + " bestKLD = " + str(round(best_kld, 3))
                )

                if mKLD < best_kld:
                    best_kld = mKLD
                    model_name = 'best_model_' + str(total_iter + 1) + '_' + str(round(best_kld, 3)) \
                                 + '_' + str(round(mSIM, 3)) \
                                 + '_' + str(round(mNSS, 3)) \
                                 + '.pth'
                    torch.save({'iter': total_iter,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()},
                               os.path.join(args.save_path, model_name))

                model.train()

            if (total_iter + 1) >= args.iters:
                exit()
