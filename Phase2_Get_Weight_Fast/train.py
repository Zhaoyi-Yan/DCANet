# config
import sys
import time
import os
import numpy as np
import torch
from config import config
from net.RES_FPN.FPN import FPN
import net.networks as networks
from eval.Estimator import Estimator
from options.train_options import TrainOptions
from Dataset.DatasetConstructor import TrainDatasetConstructor,EvalDatasetConstructor


opt = TrainOptions().parse()



# Mainly get settings for specific datasets
setting = config(opt)

log_file = os.path.join(setting.model_save_path, opt.dataset_name+'.log')
log_f = open(log_file, "w")

# Data loaders
train_dataset = TrainDatasetConstructor(
    setting.train_num,
    setting.train_img_path,
    setting.train_gt_map_path,
    mode=setting.mode,
    dataset_name=setting.dataset_name,
    device=setting.device,
    is_random_hsi=setting.is_random_hsi,
    is_flip=setting.is_flip,
    fine_size=opt.fine_size
    )
eval_dataset = EvalDatasetConstructor(
    setting.eval_num,
    setting.eval_img_path,
    setting.eval_gt_map_path,
    mode=setting.mode,
    dataset_name=setting.dataset_name,
    device=setting.device)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=setting.batch_size, num_workers=opt.nThreads)
eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=1)

# model construct
net = networks.define_net(opt.net_name)
net = networks.init_net(net, gpu_ids=opt.gpu_ids)
criterion = torch.nn.MSELoss(reduction='sum').to(setting.device) # first device is ok
estimator = Estimator(setting, eval_loader, criterion=criterion)

optimizer = networks.select_optim(net, opt)



step = 0
eval_loss, eval_mae, eval_rmse = [], [], []
for epoch_index in range(setting.epoch):
    dataset = train_dataset.shuffle()
    time_per_epoch = 0
    for train_img_index, train_img, train_gt in train_loader:
        # put data to setting.device
        train_img = train_img.to(setting.device)
        train_gt = train_gt.to(setting.device)

        net.train()
        x, y = train_img, train_gt
        start = time.time()
        prediction = net(x)
        loss = criterion(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        loss_item = loss.detach().item()
        optimizer.step()

        step += 1
        end = time.time()
        time_per_epoch += end - start

        if step % opt.print_step == 0:
            print("Step:{:d}\t, Epoch:{:d}\t, Loss:{:.4f}".format(step, epoch_index, loss_item))

    # eval
    if epoch_index % opt.eval_per_epoch == 0 and epoch_index > opt.start_eval_epoch:
        print('Evaluating step:', str(step), '\t epoch:', str(epoch_index))
        validate_MAE, validate_RMSE, validate_loss, time_cost = estimator.evaluate(net, False)
        eval_loss.append(validate_loss)
        eval_mae.append(validate_MAE)
        eval_rmse.append(eval_rmse)
        log_f.write(
            'In step {}, epoch {}, loss = {}, eval_mae = {}, eval_rmse = {}, time cost eval = {}s\n'
            .format(step, epoch_index, validate_loss, validate_MAE,
                validate_RMSE, time_cost))
        log_f.flush()
        # save model with epoch and MAE
        if setting.min_mae > validate_MAE:
            setting.min_mae = validate_MAE
            best_model_name = setting.model_save_path + "/MAE_" + str(round(validate_MAE, 2)) + \
                "_MSE_" + str(round(validate_RMSE, 2)) + '_Epoch_' + str(epoch_index) + '.pth'
            if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
                torch.save(net.module.cpu().state_dict(), best_model_name)
                net.cuda(opt.gpu_ids[0])
            else:
                torch.save(net.cpu().state_dict(), best_model_name)
