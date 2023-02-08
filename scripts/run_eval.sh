#!/bin/bash


echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash scripts/run_eval.sh DATA_PATH DATASET_TYPE DEVICE_TYPE CHECKPOINT_PATH"
echo "for example: bash scripts/run_eval.sh /path/ImageNet2012/validation imagenet2012 Ascend /path/a.ckpt "
echo "=============================================================================================================="

DATA_PATH=$1
DATASET_TYPE=$2
DEVICE_TYPE=$3
CHECKPOINT_PATH=$4

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

config_path=$(get_real_path "./${DATASET_TYPE}_config.yaml")
echo "config path is : ${config_path}"

python eval.py \
    --config_path=$config_path \
    --data_dir=$DATA_PATH \
    --dataset=$DATASET_TYPE \
    --device_target=$DEVICE_TYPE \
    --pre_trained=$CHECKPOINT_PATH > output.eval.log 2>&1 &
