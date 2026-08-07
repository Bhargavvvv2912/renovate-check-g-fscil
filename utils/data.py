"""Dataset definitions for the three supported FSCIL benchmarks.

Every benchmark exposes ``train_data`` / ``test_data`` (arrays of decoded images
or of file paths), the matching targets, and the torchvision transform lists
that :class:`utils.data_manager.DataManager` composes.

NOTE: following the original ASP / PILOT codebase, ``common_trsf`` is empty, so
images reach the ViT backbone in ``[0, 1]`` without ImageNet normalisation. Keep
this in mind when adding a new dataset -- the whole pipeline assumes it.
"""

import os

import numpy as np
from torchvision import datasets, transforms

from utils.class_names import resolve_imagenetr_class_names
from utils.toolkit import split_images_labels

DEFAULT_DATA_ROOT = "./data"


def build_transform(is_train, input_size=224):
    """Standard ViT train/test augmentation (no normalisation, see module docstring)."""
    if is_train:
        return [
            transforms.RandomResizedCrop(input_size, scale=(0.05, 1.0), ratio=(3. / 4., 4. / 3.)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]

    return [
        transforms.Resize(int((256 / 224) * input_size), interpolation=3),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
    ]


class iData(object):
    """Base class: subclasses fill in the fields below inside ``download_data``."""

    use_path = False
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None

    def __init__(self, args=None):
        self.args = args or {}
        self.data_root = self.args.get("data_root", DEFAULT_DATA_ROOT)
        self.train_trsf = build_transform(True)
        self.test_trsf = build_transform(False)
        self.common_trsf = []

    def _folder(self, *parts):
        return os.path.join(self.data_root, *parts)

    def download_data(self):
        raise NotImplementedError

    def _load_image_folders(self, name=None, root=None):
        root = root or self._folder(name)
        train_root = os.path.join(root, "train")
        test_root = os.path.join(root, "test")
        if not os.path.isdir(train_root) or not os.path.isdir(test_root):
            raise FileNotFoundError(
                "Expected ImageFolder splits at {!r} and {!r}.".format(train_root, test_root)
            )

        train_dset = datasets.ImageFolder(train_root)
        test_dset = datasets.ImageFolder(test_root)
        if train_dset.class_to_idx != test_dset.class_to_idx:
            train_only = sorted(set(train_dset.classes) - set(test_dset.classes))
            test_only = sorted(set(test_dset.classes) - set(train_dset.classes))
            raise ValueError(
                "train/test class folders do not match. train-only={} test-only={}".format(
                    train_only[:10], test_only[:10]
                )
            )
        if not train_dset.classes:
            raise RuntimeError("No class folders found under {}.".format(root))

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
        self.class_order = np.arange(len(train_dset.classes)).tolist()
        return list(train_dset.classes)


class iCIFAR224(iData):
    """CIFAR-100 resized to 224x224. Downloaded automatically."""

    def __init__(self, args=None):
        super().__init__(args)
        self.use_path = False
        self.class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100(self.data_root, train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100(self.data_root, train=False, download=True)
        self.train_data = train_dataset.data
        self.train_targets = np.array(train_dataset.targets)
        self.test_data = test_dataset.data
        self.test_targets = np.array(test_dataset.targets)


class CUB(iData):
    """CUB-200-2011, expected at ``<data_root>/cub/{train,test}/<class>/*.jpg``."""

    def __init__(self, args=None):
        super().__init__(args)
        self.use_path = True
        self.class_order = np.arange(200).tolist()

    def download_data(self):
        self._load_image_folders("cub")


class iImageNetR(iData):
    """ImageNet-R, expected at ``<data_root>/imagenet-r/{train,test}/<wnid>/*.jpg``."""

    def __init__(self, args=None):
        super().__init__(args)
        self.use_path = True
        self.class_order = np.arange(200).tolist()

    def download_data(self):
        root = self.args.get("imagenetr_root", self._folder("imagenet-r"))
        wnids = self._load_image_folders(root=root)
        classnames_txt = self.args.get(
            "imagenetr_classnames_txt", os.path.join(root, "classnames.txt")
        )
        # Stored in original ImageFolder order. DataManager later writes the
        # FSCIL class permutation, and Learner applies that permutation once.
        self.args["imagenetr_wnids"] = wnids
        self.args["class_names"] = resolve_imagenetr_class_names(wnids, classnames_txt)


DATASETS = {
    "cifar224": iCIFAR224,
    "cub": CUB,
    "imagenetr": iImageNetR,
}


def get_idata(dataset_name, args=None):
    key = str(dataset_name).lower()
    if key not in DATASETS:
        raise NotImplementedError(
            "Unknown dataset '{}'. Supported: {}.".format(dataset_name, sorted(DATASETS))
        )
    return DATASETS[key](args)
