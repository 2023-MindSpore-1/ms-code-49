#!/usr/bin/bash

model=$1
/usr/local/Ascend/atc/bin/atc \
  --model=$model \
  --framework=1 \
  --output=../data/model/vgg16 \
  --input_shape="input:1,224,224,3" \
  --enable_small_channel=1 \
  --log=error \
  --soc_version=Ascend310 \
  --insert_op_conf=aipp_vgg16_rgb.config
exit 0
