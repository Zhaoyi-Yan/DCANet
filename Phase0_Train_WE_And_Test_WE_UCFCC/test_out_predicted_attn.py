# config
import sys
import warnings
import time
import numpy as np
import torch
from config import config
from eval.Estimator_attn import Estimator
from net.networks import *
from options.test_options import TestOptions
from Dataset.DatasetConstructor import TrainDatasetConstructor,EvalDatasetConstructor

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batch_size = 1  # test code only supports batchSize = 1
opt.is_flip = 0  # no flip

#test_model_name = './output/Res_unet_aspp_1e4_b5_weight_0.25/JSTL/MAE_58.2_MSE_121.27_Epoch_273.pth'
#test_model_name = './output/Res_unet_aspp_1e4_b5_weight_0.25/JSTL/MAE_56.9_MSE_121.24_Epoch_432.pth'

# learning rate is good
#test_model_name = 'output/Phase3C_1e4_b5_attn1_fix_remove_relu/JSTL/MAE_54.53_MSE_115.56_Epoch_117.pth' # domain ok
#test_model_name = 'output/Phase3C_1e4_b5_attn1_fix_finalv/JSTL/MAE_55.31_MSE_117.8_Epoch_118.pth'  # domain ok
#test_model_name = 'output/Try2_Phase3cb_SoftHard_attn1_target_lr_1/JSTL//MAE_54.74_MSE_114.12_Epoch_85.pth' # domain ok


# Using slightly-refined DGANet+ to give different pre-attens
#test_model_name = 'output/Phase3C_1e4_b5_attn1_fix_tmp/JSTL/MAE_121.64_MSE_181.84_Epoch_4.pth'
#test_model_name = 'output/Phase3C_1e4_b5_attn1_sigmod/JSTL/MAE_55.26_MSE_117.62_Epoch_242.pth'
#test_model_name = 'output/Phase3C_1e4_b5_attn1_sigmod_tglr_100/JSTL/MAE_56.61_MSE_121.06_Epoch_171.pth'
#test_model_name = 'output/Phase3C_softmax_branchwise_1e4_b5_attn1/JSTL/MAE_57.35_MSE_120.44_Epoch_139.pth'
#test_model_name = 'output/New_Unet_softmax_branch/MAE_57.12_MSE_119.06_Epoch_93.pth'
#test_model_name = 'output/NewUnet_Phase3cb_SoftHard_attn1_target_lr_1/MAE_55.1_MSE_119.5_Epoch_67.pth'
test_model_name = 'output/New_Unet_softmax_branch/MAE_57.12_MSE_119.06_Epoch_93.pth'

# Mainly get settings for specific datasets
setting = config(opt)

# Data loaders
eval_dataset = EvalDatasetConstructor(
    setting.scene_index,
    setting.eval_num,
    setting.eval_img_path,
    setting.eval_gt_map_path,
    mode=setting.mode,
    dataset_name=setting.dataset_name,
    device=setting.device)
eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=1)

# model construct
net = define_net(opt.net_name)
net = init_net(net, gpu_ids=opt.gpu_ids)
net.module.load_state_dict(torch.load(test_model_name, map_location=str(setting.device)), False)
criterion = torch.nn.MSELoss(reduction='sum').to(setting.device)
estimator = Estimator(setting, eval_loader, criterion=criterion)


validate_MAE, validate_RMSE, validate_loss, time_cost = estimator.evaluate(net, True)
sys.stdout.write('loss = {}, eval_mae = {}, eval_rmse = {}, time cost eval = {}s\n'
                .format(validate_loss, validate_MAE, validate_RMSE, time_cost))
sys.stdout.flush()
