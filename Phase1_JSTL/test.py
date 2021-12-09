# config
import sys
import warnings
import time
import numpy as np
import torch
from config import config
from eval.Estimator import Estimator
from net.networks import *
from options.test_options import TestOptions
from Dataset.DatasetConstructor import TrainDatasetConstructor,EvalDatasetConstructor

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batch_size = 1  # test code only supports batchSize = 1
opt.is_flip = 0  # no flip

test_model_name = 'output/HRNet_relu_aspp/JSTL_SH_LgQF_NU_JU_oriImg/MAE_69.45_MSE_288.17_mae_66.05_12.21_104.91_87.49_68.11_mse_117.92_19.51_185.47_494.58_258.65_Ep_26.pth'


# Mainly get settings for specific datasets
setting = config(opt)

# Data loaders
eval_dataset = EvalDatasetConstructor(
    setting.datasets_com,
    setting.eval_img_path,
    setting.eval_gt_map_path,
    mode=setting.mode,
    dataset_name=setting.dataset_name,
    device=setting.device)
eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=1)

# model construct
net = define_net(opt)
net = init_net(net, gpu_ids=opt.gpu_ids)
net.module.load_state_dict(torch.load(test_model_name, map_location=str(setting.device)))
criterion = torch.nn.MSELoss(reduction='sum').to(setting.device)
estimator = Estimator(setting, eval_loader, criterion=criterion)

validate_MAE, validate_RMSE, validate_loss, time_cost, pred_mae, pred_mse = estimator.evaluate(net, False)
sys.stdout.write('loss = {}, eval_mae = {}, eval_rmse = {}, SHA = {}/{}, SHB = {}/{}, QNRF = {}/{}, NWPU = {}/{}, JHU = {}/{}, time cost eval = {}s\n'
                .format(validate_loss, validate_MAE, validate_RMSE, pred_mae[0], pred_mse[0], pred_mae[1], pred_mse[1], pred_mae[2], pred_mse[2], pred_mae[3], pred_mse[3], \
                         pred_mae[4], pred_mse[4], time_cost))

sys.stdout.flush()
