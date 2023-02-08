#!/bin/bash


echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_distribute_train_gpu.sh DATA_PATH"
echo "for example: bash run_distribute_train_gpu.sh /path/ImageNet2012/train"
echo "=============================================================================================================="

DATA_PATH=$1

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}
config_path=$(get_real_path "./imagenet2012_config.yaml")

mpirun -n 8 --output-filename log_output --merge-stderr-to-stdout \
  python train.py  \
    --config_path=$config_path \
    --device_target="GPU" \
    --dataset="imagenet2012" \
    --is_distributed=1 \
    --data_dir=$DATA_PATH  > output.train.log 2>&1 &
