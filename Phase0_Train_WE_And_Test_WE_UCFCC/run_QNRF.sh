#!/bin/bash
python train.py --dataset_name='QNRF' --gpu_ids='1' --optimizer='adam' --nThreads=8 --start_eval_epoch=100 --lr=1e-4 --name='Res_unet_aspp_1e4_b5' --net_name='res_unet_aspp' --batch_size=5 --eval_per_epoch=1
