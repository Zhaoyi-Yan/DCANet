#!/bin/bash
python train.py --dataset_name='WorldExpo' --gpu_ids='0' --optimizer='adam' --start_eval_epoch=300 --max_epochs=310 --lr=1e-4 --name='HRNet_relu_aspp_b16' --net_name='hrnet_aspp_relu' --batch_size=16 --eval_per_epoch=5
