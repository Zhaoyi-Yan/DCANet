#!/bin/bash

step=2
final_epoch=200
start_idx=180

count=1
j=$start_idx

while [ $j -le $final_epoch ];do
{
echo "Testing epoch : "${j}
j=$(($start_idx + $count * $step))


for scene_i in 1 2 3 4 5
do
    python test.py  --test_model_name='output/Res_unet_aspp_leaky_248_continue_noleaky/WorldExpo/Epoch_'${j}'.pth' --scene_index=${scene_i}
done

let count++

}
done
