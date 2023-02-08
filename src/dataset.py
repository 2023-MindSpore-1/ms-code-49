"""
dataset processing.
"""
import os
import multiprocessing
from PIL import Image, ImageFile
import mindspore as ms
from mindspore.common import dtype as mstype
import mindspore.dataset as de
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as vision
from src.utils.sampler import DistributedSampler

ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_num_parallel_workers(num_parallel_workers):
    """
    Get num_parallel_workers used in dataset operations.
    If num_parallel_workers > the real CPU cores number, set num_parallel_workers = the real CPU cores number.
    """
    cores = multiprocessing.cpu_count()
    if isinstance(num_parallel_workers, int):
        if cores < num_parallel_workers:
            print("The num_parallel_workers {} is set too large, now set it {}".format(num_parallel_workers, cores))
            num_parallel_workers = cores
    else:
        print("The num_parallel_workers {} is invalid, now set it {}".format(num_parallel_workers, min(cores, 8)))
        num_parallel_workers = min(cores, 8)
    return num_parallel_workers

def vgg_create_dataset(data_home, image_size, batch_size, rank_id=0, rank_size=1, training=True):
    """Data operations."""
    data_dir = os.path.join(data_home, "cifar-10-batches-bin")
    if not training:
        data_dir = os.path.join(data_home, "cifar-10-verify-bin")

    data_set = de.Cifar10Dataset(data_dir, num_shards=rank_size, shard_id=rank_id)

    rescale = 1.0 / 255.0
    shift = 0.0

    # define map operations
    random_crop_op = vision.RandomCrop((32, 32), (4, 4, 4, 4))  # padding_mode default CONSTANT
    random_horizontal_op = vision.RandomHorizontalFlip()
    resize_op = vision.Resize(image_size)  # interpolation default BILINEAR
    rescale_op = vision.Rescale(rescale, shift)
    normalize_op = vision.Normalize((0.4465, 0.4822, 0.4914), (0.2010, 0.1994, 0.2023))
    changeswap_op = vision.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    c_trans = []
    if training:
        c_trans = [random_crop_op, random_horizontal_op]
    c_trans += [resize_op, rescale_op, normalize_op,
                changeswap_op]

    # apply map operations on images
    data_set = data_set.map(operations=type_cast_op, input_columns="label")
    data_set = data_set.map(operations=c_trans, input_columns="image")

    # apply shuffle operations
    data_set = data_set.shuffle(buffer_size=10)

    # apply batch operations
    data_set = data_set.batch(batch_size=batch_size, drop_remainder=True)

    return data_set


def classification_dataset(data_dir, image_size, per_batch_size, rank=0, group_size=1,
                           mode='train',
                           input_mode='folder',
                           root='',
                           num_parallel_workers=None,
                           shuffle=None,
                           sampler=None,
                           class_indexing=None,
                           drop_remainder=True,
                           transform=None,
                           target_transform=None):
    """
    A function that returns a dataset for classification. The mode of input dataset could be "folder" or "txt".
    If it is "folder", all images within one folder have the same label. If it is "txt", all paths of images
    are written into a textfile.

    Args:
        data_dir (str): Path to the root directory that contains the dataset for "input_mode="folder"".
            Or path of the textfile that contains every image's path of the dataset.
        image_size (Union(int, sequence)): Size of the input images.
        per_batch_size (int): the batch size of evey step during training.
        rank (int): The shard ID within num_shards (default=None).
        group_size (int): Number of shards that the dataset should be divided
            into (default=None).
        mode (str): "train" or others. Default: " train".
        input_mode (str): The form of the input dataset. "folder" or "txt". Default: "folder".
        root (str): the images path for "input_mode="txt"". Default: " ".
        num_parallel_workers (int): Number of workers to read the data. Default: None.
        shuffle (bool): Whether or not to perform shuffle on the dataset
            (default=None, performs shuffle).
        sampler (Sampler): Object used to choose samples from the dataset. Default: None.
        repeat_num (int): the num of repeat dataset.
        class_indexing (dict): A str-to-int mapping from folder name to index
            (default=None, the folder names will be sorted
            alphabetically and each class will be given a
            unique index starting from 0).

    Examples:
        >>> from src.dataset import classification_dataset
        >>> # path to imagefolder directory. This directory needs to contain sub-directories which contain the images
        >>> data_dir = "/path/to/imagefolder_directory"
        >>> de_dataset = classification_dataset(data_dir, image_size=[224, 244],
        >>>                               per_batch_size=64, rank=0, group_size=4)
        >>> # Path of the textfile that contains every image's path of the dataset.
        >>> data_dir = "/path/to/dataset/images/train.txt"
        >>> images_dir = "/path/to/dataset/images"
        >>> de_dataset = classification_dataset(data_dir, image_size=[224, 244],
        >>>                               per_batch_size=64, rank=0, group_size=4,
        >>>                               input_mode="txt", root=images_dir)
    """

    # Computed from random subset of ImageNet training images
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    if transform is None:
        if mode == 'train':
            transform_img = [
                vision.RandomCropDecodeResize(image_size, scale=(0.08, 1.0)),
                vision.RandomHorizontalFlip(prob=0.5),
                vision.Normalize(mean=mean, std=std),
                vision.HWC2CHW()
            ]
        else:
            transform_img = [
                vision.Decode(),
                vision.Resize((256, 256)),
                vision.CenterCrop(image_size),
                vision.Normalize(mean=mean, std=std),
                vision.HWC2CHW()
            ]
    else:
        transform_img = transform

    if target_transform is None:
        transform_label = [C.TypeCast(mstype.int32)]
    else:
        transform_label = target_transform

    if input_mode == 'folder':
        de_dataset = de.ImageFolderDataset(data_dir, num_parallel_workers=num_parallel_workers,
                                           shuffle=shuffle, sampler=sampler, class_indexing=class_indexing,
                                           num_shards=group_size, shard_id=rank)
    else:
        dataset = TxtDataset(root, data_dir)
        sampler = DistributedSampler(dataset, rank, group_size, shuffle=shuffle)
        de_dataset = de.GeneratorDataset(dataset, ["image", "label"], sampler=sampler)

    de_dataset = de_dataset.map(operations=transform_img, input_columns="image", num_parallel_workers=8)
    de_dataset = de_dataset.map(operations=transform_label, input_columns="label", num_parallel_workers=8)

    columns_to_project = ["image", "label"]
    de_dataset = de_dataset.project(columns=columns_to_project)

    de_dataset = de_dataset.batch(per_batch_size, drop_remainder=drop_remainder)

    return de_dataset

def create_dataset(dataset_path, do_train, batch_size=32, train_image_size=224, eval_image_size=224,
                   enable_cache=False, cache_session_id=None):
    """
    create a train or eval flower dataset for vgg16

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        batch_size(int): the batch size of dataset. Default: 32
        enable_cache(bool): whether tensor caching service is used for eval. Default: False
        cache_session_id(int): If enable_cache, cache session_id need to be provided. Default: None

    Returns:
        dataset
    """
    de.config.set_prefetch_size(64)
    data_set = de.ImageFolderDataset(dataset_path, num_parallel_workers=get_num_parallel_workers(12), shuffle=True)

    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    # define map operations
    if do_train:
        trans = [
            de.vision.RandomCropDecodeResize(train_image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            de.vision.RandomHorizontalFlip(prob=0.5)
        ]
    else:
        trans = [
            de.vision.Decode(),
            de.vision.Resize(256),
            de.vision.CenterCrop(eval_image_size)
        ]
    trans_norm = [de.vision.Normalize(mean=mean, std=std), de.vision.HWC2CHW()]

    type_cast_op = de.transforms.TypeCast(ms.int32)
    trans_work_num = 24
    data_set = data_set.map(operations=trans, input_columns="image",
                            num_parallel_workers=get_num_parallel_workers(trans_work_num))
    data_set = data_set.map(operations=trans_norm, input_columns="image",
                            num_parallel_workers=get_num_parallel_workers(12))
    # only enable cache for eval
    if do_train:
        enable_cache = False
    if enable_cache:
        if not cache_session_id:
            raise ValueError("A cache session_id must be provided to use cache.")
        eval_cache = de.DatasetCache(session_id=int(cache_session_id), size=0)
        data_set = data_set.map(operations=type_cast_op, input_columns="label",
                                num_parallel_workers=get_num_parallel_workers(12),
                                cache=eval_cache)
    else:
        data_set = data_set.map(operations=type_cast_op, input_columns="label",
                                num_parallel_workers=get_num_parallel_workers(12))

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)

    return data_set


class TxtDataset:
    """
    create txt dataset.

    Args:
    Returns:
        de_dataset.
    """
    def __init__(self, root, txt_name):
        super(TxtDataset, self).__init__()
        self.imgs = []
        self.labels = []
        fin = open(txt_name, "r")
        for line in fin:
            img_name, label = line.strip().split(' ')
            self.imgs.append(os.path.join(root, img_name))
            self.labels.append(int(label))
        fin.close()

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert('RGB')
        return img, self.labels[index]

    def __len__(self):
        return len(self.imgs)