#!/bin/bash
python train.py --dataset_name='JSTL_large' --gpu_ids='0,1' --optimizer='adam' --start_eval_epoch=-1 --eval_per_epoch=2 --lr=5e-5 --name='HRNet_relu_aspp' --net_name='hrnet_aspp_relu' --batch_size=8 --nThreads=8
