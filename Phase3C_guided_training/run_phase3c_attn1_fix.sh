#!/bin/bash
python train.py --dataset_name='JSTL_large' --gpu_ids='0,1' --optimizer='adam' \
               --start_eval_epoch=-1 --nThreads=8 --batch_size=8 --lr=1e-5 --continue_train=1 \
               --fix_domain=1 --name='Phase3C_b16_attn1_69.45' --loss_attn_weight=1  \
               --load_model_name='output/JSTL_large/MAE_54.97_MSE_117.73_mae_61.12_9.27_94.86_mse_107.84_16.32_167.15_Ep_258.pth' \
               --domain_center_model='average_clip_domain_center_54.97.npz' \
               --net_name='hrnet_aspp_relu' --eval_per_epoch=1
