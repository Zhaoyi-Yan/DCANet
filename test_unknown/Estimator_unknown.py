'''
It is derived from 'Estimated_slow', aiming to evaluate single unknown image.
'''
import random
import math
import os
import numpy as np
import sys
from PIL import Image
from metrics import AEBatch, SEBatch
import time
import torch
import scipy.io as scio
import util.utils as util

class Estimator(object):
    def __init__(self, setting, eval_loader, criterion=torch.nn.MSELoss(reduction="sum")):
        self.setting = setting
        self.ae_batch = AEBatch().to(self.setting.device)
        self.se_batch = SEBatch().to(self.setting.device)
        self.criterion = criterion
        self.eval_loader = eval_loader

    def evaluate(self, net, is_save=False):
        net.eval()
        pred_counts = []

        cur, time_cost = 0, 0
        for eval_resized, eval_img_path, eval_img, eval_img_shape in self.eval_loader: # eval_resized is not 9-patched
            eval_img_shape = eval_img_shape[0]
            eval_img = eval_img.to(self.setting.device)

            start = time.time()
            eval_patchs = torch.squeeze(eval_img)
            
            prediction_map = torch.zeros(1, 1, eval_img_shape[1]//2, eval_img_shape[2]//2).to(self.setting.device)
            eval_img_path = eval_img_path[0]

            with torch.no_grad():
                eval_prediction = net(eval_patchs, is_eval=True)
                eval_patchs_shape = eval_prediction.shape
                # test cropped patches
                self.test_crops(eval_patchs_shape, eval_prediction, prediction_map)
                pred_count = torch.sum(prediction_map).item()
                pred_counts.append(pred_count)

                if is_save:
                   # validate_pred_map = np.squeeze(prediction_map.permute(0, 2, 3, 1).data.cpu().numpy())
                   # pred_cnt = np.sum(validate_pred_map)
                    util.save_image(util.tensor2im(prediction_map), os.path.join('out_imgs', os.path.splitext(eval_img_path.split('/')[-1])[0]+'_pred.png'))
                    eval_resized = (np.transpose(eval_resized[0].data.cpu().float().numpy(), (1, 2, 0)) * 255).astype(np.uint8)
                    util.save_image(eval_resized, os.path.join('out_imgs', os.path.splitext(eval_img_path.split('/')[-1])[0]+'_ori.png'))
                    with open(os.path.join('out_imgs', os.path.splitext(eval_img_path.split('/')[-1])[0]+'.txt'), "w") as f:
                        f.write(str(pred_count))
                        f.write('\n')


            cur += 1
            torch.cuda.synchronize()
            end = time.time()
            time_cost += (end - start)

        return pred_counts

    # New Function
    def get_gt_num(self, eval_img_path):
        mat_name = eval_img_path.replace('images', 'ground_truth')[:-4] + ".mat"
        gt_counts = len(scio.loadmat(mat_name)['annPoints'])

        return gt_counts

    def test_crops(self, eval_shape, eval_p, pred_m):
        for i in range(3):
            for j in range(3):
                start_h, start_w = math.floor(eval_shape[2] / 4), math.floor(eval_shape[3] / 4)
                valid_h, valid_w = eval_shape[2] // 2, eval_shape[3] // 2
                pred_h = math.floor(3 * eval_shape[2] / 4) + (eval_shape[2] // 2) * (i - 1)
                pred_w = math.floor(3 * eval_shape[3] / 4) + (eval_shape[3] // 2) * (j - 1)
                if i == 0:
                    valid_h = math.floor(3 * eval_shape[2] / 4)
                    start_h = 0
                    pred_h = 0
                elif i == 2:
                    valid_h = math.ceil(3 * eval_shape[2] / 4)

                if j == 0:
                    valid_w = math.floor(3 * eval_shape[3] / 4)
                    start_w = 0
                    pred_w = 0
                elif j == 2:
                    valid_w = math.ceil(3 * eval_shape[3] / 4)
                pred_m[:, :, pred_h:pred_h + valid_h, pred_w:pred_w + valid_w] += eval_p[i * 3 + j:i * 3 + j + 1, :,start_h:start_h + valid_h, start_w:start_w + valid_w]
