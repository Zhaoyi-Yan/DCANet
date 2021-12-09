#!/bin/bash
python train.py --dataset_name='SHA' --gpu_ids='1' --optimizer='adamW' --start_eval_epoch=100 --lr=1e-4 --name='Res_unet_1e4_b8_adamW' --net_name='res_unet' --batch_size=8 --eval_per_epoch=1
