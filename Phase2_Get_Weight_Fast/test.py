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

test_model_name = 'output/HRNet_relu_aspp/JSTL_large/MAE_54.97_MSE_117.73_mae_61.12_9.27_94.86_mse_107.84_16.32_167.15_Ep_258.pth'




save_model_MAE  = 'mae54.97_'


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
net = define_net(opt.net_name)
net = init_net(net, gpu_ids=opt.gpu_ids)
net.module.load_state_dict(torch.load(test_model_name, map_location=str(setting.device)))
criterion = torch.nn.MSELoss(reduction='sum').to(setting.device)
estimator = Estimator(setting, eval_loader, criterion=criterion)
optimizer = select_optim(net, opt)

validate_MAE, validate_RMSE, validate_loss, time_cost = estimator.evaluate(net, weight_scale=opt.weight_scale, save_model_MAE=save_model_MAE, optimizer=optimizer) # I add weight_scale here
sys.stdout.write('loss = {}, eval_mae = {}, eval_rmse = {}, time cost eval = {}s\n'
                .format(validate_loss, validate_MAE, validate_RMSE, time_cost))
sys.stdout.flush()
