"""export checkpoint file into models"""
import os
import numpy as np

from mindspore import Tensor, context
import mindspore.common.dtype as mstype
from mindspore.train.serialization import load_checkpoint, export

from src.vgg import vgg16

from model_utils.moxing_adapter import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id


def modelarts_pre_process():
    '''modelarts pre process function.'''
    config.file_name = os.path.join(config.output_path, config.file_name)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_export():
    config.image_size = list(map(int, config.image_size.split(',')))

    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    if config.device_target == "Ascend":
        config.device_id = get_device_id()
        context.set_context(device_id=config.device_id)

    if config.dataset == "cifar10":
        net = vgg16(num_classes=config.num_classes, args=config)
    else:
        net = vgg16(config.num_classes, config, phase="test")

    load_checkpoint(config.ckpt_file, net=net)
    net.set_train(False)

    input_data = Tensor(np.zeros([config.batch_size, 3, config.image_size[0], config.image_size[1]]), mstype.float32)
    export(net, input_data, file_name=config.file_name, file_format=config.file_format)


if __name__ == '__main__':
    run_export()
