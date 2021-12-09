#!/bin/bash
python train_fast.py --dataset_name='JSTL_large' --gpu_ids='0,1' --optimizer='adam' \
               --start_eval_epoch=-1 --nThreads=8 --batch_size=8 --lr=1e-5 --continue_train=1 \
               --fix_domain=1 --name='Phase3C_b16_attn1_56.4' --loss_attn_weight=1 --base_mae='62.71,9.06,97.74' \
               --load_model_name='output/JSTL_large/MAE_56.4_MSE_117.87_mae_62.71_9.06_97.74_mse_110.88_16.41_166.3_Ep_257.pth' \
               --domain_center_model='average_clip_domain_center_56.4.npz' \
               --net_name='hrnet_aspp_relu' --eval_per_epoch=1
