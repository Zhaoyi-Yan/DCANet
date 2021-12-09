import random
import os
import math
import numpy as np
import sys
from PIL import Image
from util.utils import show
from metrics import AEBatch, SEBatch
import time
import net.networks as networks
import torch
import scipy.io as scio
from torch.autograd import grad

class Estimator(object):
    def __init__(self, setting, eval_loader, criterion=torch.nn.MSELoss(reduction="sum")):
        self.datasets_com = setting.datasets_com
        self.setting = setting
        self.ae_batch = AEBatch().to(self.setting.device)
        self.se_batch = SEBatch().to(self.setting.device)
        self.criterion = criterion
        self.eval_loader = eval_loader

    def nth_derivative(self, f, wrt, n):
        for i in range(n):
            grads = grad(f, wrt, create_graph=True, allow_unused=True)[0]
            f = grads.sum()
        return grads

    def evaluate(self, net, weight_scale=0.0, save_model_MAE='50', optimizer=None):
        net.train()
        total_channels = 4*128 # 4*128
        MAE_, MSE_, loss_, pred_ = [], [], [], []
        cur, time_cost = 0, 0
        for eval_img_path, eval_ori_tensor, eval_img, eval_gt, class_id in self.eval_loader:
            print(eval_img_path)
            tmpMAE_, tmpMSE_, tmploss_ = [], [], []
            eval_img = eval_img.to(self.setting.device)
            eval_gt = eval_gt.to(self.setting.device)

            start = time.time()
            eval_patchs = eval_img
          #  eval_patchs = torch.squeeze(eval_img)
            eval_gt_shape = eval_gt.shape
            eval_img_path = eval_img_path[0]

            #********************* It is newly added ***************************
            # -1: normal performace
            # i: only zero the channel of index i
          #  for zero_index in np.arange(-1, total_channels, 1):
            eval_prediction, detached = net(eval_patchs, zero_index=0, weight_scale=0)
            eval_patchs_shape = eval_prediction.shape

            # test cropped patches
          #  prediction_map = torch.zeros_like(eval_gt, requires_grad=True) # zero out the prediction_map before test_crops
          #  self.test_crops(eval_patchs_shape, eval_prediction, prediction_map)
            gt_counts = self.get_gt_num(eval_img_path)

            # the output of the net is precisely the whole map
            prediction_map = eval_prediction
            pred_counts = torch.sum(prediction_map).cpu().item()
            # calculate metrics
            batch_ae = self.ae_batch(prediction_map, gt_counts).detach().cpu().numpy()
            batch_se = self.se_batch(prediction_map, gt_counts).detach().cpu().numpy()
            loss = torch.abs(prediction_map.sum() - eval_gt.sum()) # using eval_gt
          #  loss = self.criterion.forward(prediction_map, eval_gt)

            optimizer.zero_grad()

            att = np.zeros(512)
            att_tmp = self.nth_derivative(f=loss, wrt=detached, n=1)

            att = - att_tmp.detach().cpu().numpy()[0] * detached.detach().cpu().numpy()[0]
            optimizer.zero_grad()

            att_tmp = self.nth_derivative(f=loss, wrt=detached, n=2)
            att = att + 0.5*att_tmp.detach().cpu().numpy()[0] * (detached.detach().cpu().numpy()[0] ** 2)
            optimizer.zero_grad()

            delta_loss = np.sum(att, axis=(1,2))

            channel_weight_path = os.path.join('fast_out_oriImg', os.path.splitext(eval_img_path.split('/')[-1])[0]) + '.npz'
            np.savez(channel_weight_path, delta_loss = delta_loss)
            torch.cuda.synchronize()
            end = time.time()
            time_cost += (end - start)
            cur += 1

        # return the validate loss, validate MAE and validate RMSE
 #       MAE_, MSE_, loss_ = np.reshape(MAE_, [-1]), np.reshape(MSE_, [-1]), np.reshape(loss_, [-1])
        return np.mean(MAE_), np.sqrt(np.mean(MSE_)), np.mean(loss_), time_cost

    def get_cur_dataset(self, img_name):
        check_list = [img_name.find(da) for da in self.datasets_com]
        check_list = np.array(check_list)
        cur_idx = np.where(check_list != -1)[0][0]
        return self.datasets_com[cur_idx]

    def get_gt_num(self, eval_img_path):
        mat_name = eval_img_path.replace('images', 'ground_truth')[:-4] + ".mat"
        gt_counts = len(scio.loadmat(mat_name)['annPoints'])

        return gt_counts

    # For JSTL, this function is not supported
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
