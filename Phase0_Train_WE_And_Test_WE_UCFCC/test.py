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

#test_model_name = 'output/Res_unet_aspp_leaky/WorldExpo/MAE_5.26_MSE_6.25_Epoch_166.pth'
#test_model_name = 'output/HRNet_relu_aspp_b16/JSTL/MAE_54.89_MSE_122.7_mae_59.64_10.16_94.61_mse_99.23_17.83_178.42_Ep_345.pth'
#test_model_name = 'output/HRNet_relu_aspp_b16/JSTL/MAE_54.95_MSE_124.66_mae_63.86_9.0_93.58_mse_107.3_16.26_179.4_Ep_321.pth'
#test_model_name = 'output/HRNet_relu_aspp_b16/JSTL/MAE_55.37_MSE_113.35_mae_63.89_11.17_92.54_mse_104.3_17.0_160.64_Ep_257.pth'
#test_model_name = 'output/HRNet_relu_aspp_b16/JSTL/MAE_55.02_MSE_117.63_mae_62.98_10.75_92.55_mse_101.18_16.22_169.24_Ep_271.pth'
#test_model_name = 'output/HRNet_relu_aspp_b16/WorldExpo/Epoch_260.pth'
#test_model_name = 'output/Phase3C_relu_aspp/MAE_53.53_MSE_114.62_mae_59.01_7.87_93.75_mse_99.24_12.86_164.94_Ep_339.pth'
#test_model_name = 'MAE_51.16_MSE_111.85_mae_58.25_7.24_88.86_mse_99.28_11.81_160.18_Ep_63.pth'
#test_model_name = 'MAE_50.16_MSE_105.04_mae_56.09_8.02_86.8_mse_95.67_12.97_149.46_Ep_107.pth'
#test_model_name = 'MAE_58.29_MSE_122.42_mae_68.59_9.2_99.12_mse_110.15_14.02_174.74_Ep_3.pth'
#test_model_name = 'MAE_55.47_MSE_114.94_mae_66.61_10.22_92.22_mse_103.44_15.61_163.86_Ep_162.pth'
#test_model_name = 'MAE_54.17_MSE_118.13_mae_63.11_7.22_93.72_mse_99.47_12.33_170.96_Ep_93.pth'
#test_model_name = 'MAE_55.88_MSE_120.72_mae_60.72_9.24_97.38_mse_96.75_15.47_176.01_Ep_132.pth'
#test_model_name = 'output/HRNet_relu_aspp_b16/WorldExpo/Epoch_305.pth'

#test_model_name = 'MAE_50.25_MSE_107.48_mae_57.98_7.19_86.79_mse_100.78_11.57_152.04_Ep_96.pth' # for ldg 9-7
#test_model_name = 'MAE_50.57_MSE_107.11_mae_56.98_7.13_88.17_mse_96.22_11.32_153.01_Ep_134.pth'
#test_model_name = 'MAE_51.59_MSE_106.27_mae_58.22_7.14_90.04_mse_97.04_11.61_151.24_Ep_75.pth'
#test_model_name = 'MAE_53.92_MSE_114.4_mae_57.46_9.25_94.27_mse_97.36_15.49_164.95_Ep_302.pth'
#test_model_name = 'MAE_54.89_MSE_122.7_mae_59.64_10.16_94.61_mse_99.23_17.83_178.42_Ep_345.pth'
#test_model_name = 'MAE_56.0_MSE_174.11_mae_46.56_9.61_88.45_54.33_59.98_mse_73.85_14.82_147.09_198.42_195.11_Ep_56.pth'

test_model_name = 'MAE_64.98_MSE_280.77_mae_60.21_9.2_90.17_69.25_69.95_mse_108.73_14.08_153.78_458.88_268.95_Ep_60.pth'


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

net.module.load_state_dict(torch.load(test_model_name, map_location=str(setting.device)), True)
criterion = torch.nn.MSELoss(reduction='sum').to(setting.device)
estimator = Estimator(setting, eval_loader, criterion=criterion)


validate_MAE, validate_RMSE, validate_loss, time_cost = estimator.evaluate(net, False)
sys.stdout.write('loss = {}, eval_mae = {}, eval_rmse = {}, time cost eval = {}s\n'
                .format(validate_loss, validate_MAE, validate_RMSE, time_cost))
sys.stdout.flush()
