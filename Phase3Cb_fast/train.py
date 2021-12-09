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
    setting.datasets_com,
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
    setting.datasets_com,
    setting.eval_img_path,
    setting.eval_gt_map_path,
    mode=setting.mode,
    dataset_name=setting.dataset_name,
    device=setting.device)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=setting.batch_size, shuffle=True, num_workers=opt.nThreads)
eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=1, shuffle=True)

# model construct
net = networks.define_net(opt)
net = networks.init_net(net, gpu_ids=opt.gpu_ids)


print('Loading pretrained model')
print('---------------------')
net.module.load_state_dict(torch.load(opt.load_model_name, map_location=str(setting.device)), strict=False) # using False here


criterion = torch.nn.MSELoss(reduction='sum').to(setting.device) # first device is ok
criterion2 = torch.nn.MSELoss(reduction='sum').to(setting.device) # first device is ok
estimator = Estimator(setting, eval_loader, criterion=criterion)
optimizer = networks.select_optim(net, opt)

step = 0
eval_loss, eval_mae, eval_rmse = [], [], []

high_pred_mae = [1e6, 1e6, 1e6]

for epoch_index in range(setting.epoch):
    time_per_epoch = 0
    net.train()
    for train_img, train_gt, class_id in train_loader:
        # put data to setting.device
        train_img = train_img.to(setting.device)
        train_gt = train_gt.to(setting.device)
        domain = class_id.view(-1)

        x, y = train_img, train_gt
        start = time.time()
        prediction, pred_attn, target_attn = net(x)
        loss = criterion(prediction, y)
        loss_attn = criterion2(pred_attn, target_attn)
        optimizer.zero_grad()
        (loss + loss_attn).backward()
        loss_item = loss.detach().item()
        loss_attn_item = loss_attn.detach().item()
        optimizer.step()

        step += 1
        end = time.time()
        time_per_epoch += end - start

        if step % opt.print_step == 0:
            print("Step:{:d}\t, Epoch:{:d}\t, Loss:{:.4f}, Attn:{:.4f}".format(step, epoch_index, loss_item, loss_attn_item))

    # eval
    if epoch_index % opt.eval_per_epoch == 0 and epoch_index > opt.start_eval_epoch:
        print('Evaluating step:', str(step), '\t epoch:', str(epoch_index))
        validate_MAE, validate_RMSE, validate_loss, time_cost, pred_mae, pred_mse = estimator.evaluate(net, False) # pred_mae and pred_mse are for seperate datasets
        eval_loss.append(validate_loss)
        eval_mae.append(validate_MAE)
        eval_rmse.append(eval_rmse)
        log_f.write(
            'In step {}, epoch {}, loss = {}, eval_mae = {}, eval_rmse = {}, mae_sep = {},{},{}, mse_sep = {},{},{}, time cost eval = {}s\n'
            .format(step, epoch_index, validate_loss, validate_MAE, validate_RMSE, pred_mae[0], pred_mae[1], pred_mae[2],
                    pred_mse[0], pred_mse[1], pred_mse[2], time_cost))
        log_f.flush()
        # save model with epoch and MAE

        # Two kinds of conditions, we save models
        save_now = False
        if setting.min_mae >= validate_MAE:
            setting.min_mae = validate_MAE
            save_now = True

        if pred_mae[0] < high_pred_mae[0] or pred_mae[1] < high_pred_mae[1] or pred_mae[2] < high_pred_mae[2]:
            high_pred_mae[0] = pred_mae[0]
            high_pred_mae[1] = pred_mae[1]
            high_pred_mae[2] = pred_mae[2]
            save_now = True

        if save_now:
            best_model_name = setting.model_save_path + "/MAE_" + str(round(validate_MAE, 2)) + \
                "_MSE_" + str(round(validate_RMSE, 2)) + '_mae_' + str(round(pred_mae[0], 2)) + \
                '_' + str(round(pred_mae[1], 2)) + '_' + str(round(pred_mae[2], 2)) + \
                '_mse_' + str(round(pred_mse[0], 2)) + \
                '_' + str(round(pred_mse[1], 2)) + '_' + str(round(pred_mse[2], 2))  + \
                '_Ep_' + str(epoch_index) + '.pth'
            if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
                torch.save(net.module.cpu().state_dict(), best_model_name)
                net.cuda(opt.gpu_ids[0])
            else:
                torch.save(net.cpu().state_dict(), best_model_name)


if __name__ == '__main__':
    main()