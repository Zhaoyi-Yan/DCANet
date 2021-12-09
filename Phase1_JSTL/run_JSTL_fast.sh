#!/bin/bash
python3 train_fast.py --load_model_name='' --dataset_name='JSTL_large' --gpu_ids='0,1' --optimizer='adam' --start_eval_epoch=150 --eval_per_epoch=1 --lr=5e-5 --name='HRNet_relu_aspp' --net_name='hrnet_relu_aspp' --batch_size=16 --nThreads=8
