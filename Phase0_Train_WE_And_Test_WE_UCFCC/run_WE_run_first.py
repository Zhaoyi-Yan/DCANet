#!/bin/bash
python train.py --dataset_name='WorldExpo' --gpu_ids='0' --optimizer='adam' --start_eval_epoch=190 --max_epochs=200 --lr=1e-4 --name='H' --net_name='res_unet_aspp_leaky' --batch_size=16 --eval_per_epoch=2
