'''
It is derived from 'Estimated_slow', aiming to evaluate single unknown image.
'''
import random
import math
import os
import numpy as np
import sys
import torch.nn.functional as F
from PIL import Image
from metrics import AEBatch, SEBatch
import time
import torch
import scipy.io as scio
import util.utils as util
import cv2

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
        gt_counts = []
        gap_counts = []

        cur, time_cost = 0, 0
        for eval_resized, eval_img_path, eval_img, eval_img_shape in self.eval_loader: # eval_resized is not 9-patched
            print(eval_img_path)
            # gt_count = gt_count[0]
            eval_img_path = eval_img_path[0]
            gt_count = self.get_gt_num(eval_img_path)
            eval_img_shape = eval_img_shape[0]
            eval_img = eval_img.to(self.setting.device)

            start = time.time()
            eval_patchs = torch.squeeze(eval_img)
            
            prediction_map = torch.zeros(1, 1, eval_img_shape[1]//2, eval_img_shape[2]//2).to(self.setting.device)


            with torch.no_grad():
                eval_prediction = net(eval_patchs, is_eval=True)
                eval_patchs_shape = eval_prediction.shape
                # test cropped patches
                self.test_crops(eval_patchs_shape, eval_prediction, prediction_map)
                pred_count = torch.sum(prediction_map).item()
                pred_counts.append(pred_count)
                gt_counts.append(gt_count)
                gap_counts.append(abs(pred_count - gt_count))

                if is_save:
                    util.save_image(util.tensor2im(prediction_map), os.path.join('out_imgs', os.path.splitext(eval_img_path.split('/')[-1])[0]+'_pred.png'))
                    with open(os.path.join('out_imgs', os.path.splitext(eval_img_path.split('/')[-1])[0]+'.txt'), "w") as f:
                        f.write(str(pred_count))
                        f.write('\n')
                    
                    h, w = eval_resized.shape[-2], eval_resized.shape[-1]
                    # New method
                    eval_resized = (np.transpose(eval_resized[0].data.cpu().float().numpy(), (1, 2, 0)) * 255).astype(np.uint8)
                    eval_resized_cv = cv2.cvtColor(eval_resized, cv2.COLOR_RGB2BGR)
                    # cv2.rectangle(eval_resized_cv, (h//30, h//30), (h//5, h//8), (33, 140, 255), 2)
                    prediction_map = F.interpolate(prediction_map, scale_factor=2 , mode='bilinear', align_corners=False)
                    vis_img = prediction_map[0].repeat(3, 1, 1).permute(1, 2, 0).cpu().float().numpy()
                    vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-5)
                    vis_img = (vis_img * 255).astype(np.uint8)
                    vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
                    # merge perdiction map and eval_img
                    merged_resized_cv = np.concatenate((eval_resized_cv, vis_img), axis=0)
                    eval_resized_cv = merged_resized_cv

                    # model_text = "Model: PFDNet_DCANet"
                    pred_text = "Pred:  "+str(round(pred_count, 2))
                    gt_text   = "GT:    "+str(round(gt_count, 2))
                    gap_text  = "Error: "+str(round(abs(pred_count - gt_count), 2))
                    avg_gap_text  = "Avg Error: "+str(round(np.mean(np.array(gap_counts)), 2))
                    # cv2.putText(eval_resized_cv, model_text, (w//4, h//10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2) # the first item, horizontal
                    cv2.putText(eval_resized_cv, gt_text, (h//20, h//15), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)                    
                    cv2.putText(eval_resized_cv, pred_text, (h//20, h//10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                    cv2.putText(eval_resized_cv, gap_text, (h//20, h//7), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                    cv2.putText(eval_resized_cv, avg_gap_text, (w - w//4, h//12), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)                  
                    cv2.imwrite(os.path.join('out_imgs_add_den', os.path.splitext(eval_img_path.split('/')[-1])[0]+'_ori.png'), eval_resized_cv)

                    # Ori navie method
                    # eval_resized = (np.transpose(eval_resized[0].data.cpu().float().numpy(), (1, 2, 0)) * 255).astype(np.uint8)
                    # util.save_image(eval_resized, os.path.join('out_imgs', os.path.splitext(eval_img_path.split('/')[-1])[0]+'_ori.png'))



            cur += 1
            torch.cuda.synchronize()
            end = time.time()
            time_cost += (end - start)

        return pred_counts, np.mean(np.array(gap_counts))

    # New Function
    def get_gt_num(self, eval_img_path):
        mat_name = eval_img_path.replace('_img', '_mat')[:-4] + ".mat"
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
