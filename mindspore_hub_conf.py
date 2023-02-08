"""hub config."""
from src.vgg import vgg16 as VGG16
from model_utils.config import get_config_static

def vgg16(*args, **kwargs):
    return VGG16(*args, **kwargs)


def create_network(name, *args, **kwargs):
    if name == "vgg16":
        num_classes = kwargs.get("num_classes", 10)
        if "num_classes" in kwargs:
            del kwargs["num_classes"]
        if num_classes == 10:
            config = get_config_static(config_path="../cifar10_config.yaml")
        elif num_classes == 1000:
            config = get_config_static(config_path="../imagenet2012_config.yaml")
        return vgg16(num_classes=num_classes, args=config, *args, **kwargs)
    raise NotImplementedError(f"{name} is not implemented in the repo")
