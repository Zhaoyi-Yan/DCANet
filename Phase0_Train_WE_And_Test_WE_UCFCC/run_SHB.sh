#!/bin/bash
python train.py --dataset_name='SHB' --gpu_ids='1' --nThreads=4 --net_name='res_unet_aspp_leaky' --optimizer='adam' --start_eval_epoch=100 --batch_size=5 --lr=1e-4 --name='Res_unet_aspp_1e4_b5' --eval_per_epoch=1
