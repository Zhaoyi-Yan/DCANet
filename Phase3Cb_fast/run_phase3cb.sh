#!/bin/bash
python train_fast.py --dataset_name='JSTL_large' --gpu_ids='0,1' --optimizer='adam' \
               --start_eval_epoch=-1 --nThreads=8 --batch_size=8 --lr=5e-7 --continue_train=1 \
               --fix_domain=0 --name='Phase3Cb_56.4_again' --target_lr=0.008 --loss_attn_weight=1  --base_mae='61.66,8.52,90.95' \
               --load_model_name='output/Phase3C_b16_attn1_56.4/MAE_53.24_MSE_108.55_mae_61.66_8.52_90.95_mse_102.8_13.79_153.02_Ep_21.pth' \
               --domain_center_model='average_clip_domain_center_56.4.npz' \
               --net_name='hrnet_aspp_relu' --eval_per_epoch=1
