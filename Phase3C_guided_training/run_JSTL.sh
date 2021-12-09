#!/bin/bash
python train.py --dataset_name='JSTL' --gpu_ids='0' --optimizer='adam' --start_eval_epoch=250 --lr=1e-4 --name='HRNet_relu_aspp_b16' --net_name='hrnet_aspp_relu' --batch_size=16 --nThreads=16 --eval_per_epoch=1
