import random
import h5py
import os
import math
import numpy as np
import sys
from PIL import Image
from util.utils import show
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
        MAE_, MSE_, loss_ = [], [], []
        rand_number, cur, time_cost = random.randint(0, self.setting.eval_num - 1), 0, 0
        for eval_img_path, eval_ori_img, eval_img_index, eval_img, eval_gt in self.eval_loader:
            eval_img = eval_img.to(self.setting.device)
            eval_gt = eval_gt.to(self.setting.device)

            start = time.time()
            eval_patchs = torch.squeeze(eval_img)
            eval_gt_shape = eval_gt.shape
            prediction_map = torch.zeros_like(eval_gt)
            img_index = eval_img_index.cpu().numpy()[0]
            eval_img_path = eval_img_path[0]

            with torch.no_grad():
               # eval_prediction = net(eval_patchs)
                eval_prediction, attn = net(eval_patchs)
                eval_patchs_shape = eval_prediction.shape
                # test cropped patches
                self.test_crops(eval_patchs_shape, eval_prediction, prediction_map)
                gt_counts = self.get_gt_num(img_index)
                # calculate metrics
                batch_ae = self.ae_batch(prediction_map, gt_counts).data.cpu().numpy()
                batch_se = self.se_batch(prediction_map, gt_counts).data.cpu().numpy()
                loss = self.criterion(prediction_map, eval_gt)
                loss_.append(loss.data.item())
                MAE_.append(batch_ae)
                MSE_.append(batch_se)
                # save
                if is_save:
                    attn = torch.mean(attn, dim=0).view(128, 1)
                    np.save(os.path.join("attn_npys", os.path.splitext(eval_img_path.split('/')[-1])[0]+'.npy'), attn.cpu().numpy())

            cur += 1
            torch.cuda.synchronize()
            end = time.time()
            time_cost += (end - start)

        # return the validate loss, validate MAE and validate RMSE
        MAE_, MSE_, loss_ = np.reshape(MAE_, [-1]), np.reshape(MSE_, [-1]), np.reshape(loss_, [-1])
        return np.mean(MAE_), np.sqrt(np.mean(MSE_)), np.mean(loss_), time_cost

    def get_gt_num(self, index):
        if self.setting.dataset_name == "QNRF":
            gt_path = self.setting.eval_gt_path + "/img_" + ("%04d" % (index)) + "_ann.mat"
            gt_counts = len(scio.loadmat(gt_path)['annPoints'])
        elif self.setting.dataset_name == "SHA" or self.setting.dataset_name == "SHB":
            gt_path = self.setting.eval_gt_path + "/GT_IMG_" + str(index) + ".mat"
            gt_counts = len(scio.loadmat(gt_path)['image_info'][0][0][0][0][0])
        elif self.setting.dataset_name == "UCF50":
            test_list = []
            if self.setting.scene_index == 1:
                test_list = [1, 2, 11, 19, 20, 21, 25, 33, 48, 50]
            elif self.setting.scene_index == 2:
                test_list = [9, 10, 16, 18, 26, 27, 30, 40, 44, 47]
            elif self.setting.scene_index == 3:
                test_list = [5, 13, 17, 22, 31, 38, 41, 42, 45, 49]
            elif self.setting.scene_index == 4:
                test_list = [4, 6, 8, 14, 23, 28, 29, 34, 37, 39]
            elif self.setting.scene_index == 5:
                test_list = [3, 7, 12, 15, 24, 32, 35, 36, 43, 46]
            else:
                raise ValueError('...')

            gt_path = self.setting.eval_gt_path + "/" + str(test_list[index-1]) + "_ann.mat" # index starts from 1.
            gt_counts = h5py.File(gt_path)['annPoints'].value.astype(np.float32).shape[1]
        else:
            raise NameError("No such dataset, only support SHA, SHB, QNRF")
        return gt_counts


    def show_sample(self, index, gt_counts, pred_counts, eval_gt_map, eval_pred_map):
        if self.setting.dataset_name == "QNRF":
            origin_image = Image.open(self.setting.eval_img_path + "/img_" + ("%04d" % index) + ".jpg")
        elif self.setting.dataset_name == "SHA" or self.setting.dataset_name == "SHB":
            origin_image = Image.open(self.setting.eval_img_path + "/IMG_" + str(index) + ".jpg")
        else:
            raise NameError("No such dataset, only support SHA, SHB, QNRF")
        show(origin_image, eval_gt_map, eval_pred_map, index)
        sys.stdout.write('The gt counts of the above sample:{}, and the pred counts:{}\n'.format(gt_counts, pred_counts))

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
