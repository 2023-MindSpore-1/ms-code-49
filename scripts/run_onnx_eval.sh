#!/bin/bash


echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash scripts/run_onnx_eval.sh DATA_PATH DATASET_TYPE DEVICE_TYPE ONNX_MODEL_PATH"
echo "for example: bash scripts/run_onnx_eval.sh /path/ImageNet2012/validation imagenet2012 GPU /path/a.onnx "
echo "=============================================================================================================="

DATA_PATH=$1
DATASET_TYPE=$2
DEVICE_TYPE=$3
ONNX_MODEL_PATH=$4

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

config_path=$(get_real_path "./${DATASET_TYPE}_config.yaml")
echo "config path is : ${config_path}"

python eval_onnx.py \
    --config_path=$config_path \
    --data_dir=$DATA_PATH \
    --dataset=$DATASET_TYPE \
    --device_target=$DEVICE_TYPE \
    --file_name=$ONNX_MODEL_PATH > output.eval_onnx.log 2>&1 &
