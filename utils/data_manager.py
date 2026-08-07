"""Task splitting, class ordering and few-shot sampling for FSCIL."""

import logging
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.data import get_idata


class DataManager(object):
    """Splits a dataset into one base task plus a sequence of k-shot tasks.

    ``_increments[0] = init_cls`` and every later entry equals ``increment``;
    a shorter final task is appended when the class count is not divisible.
    """

    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, args):
        self.args = args
        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed)

        assert init_cls <= len(self._class_order), "Not enough classes."
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

    @property
    def nb_tasks(self):
        return len(self._increments)

    @property
    def nb_classes(self):
        return len(self._class_order)

    def get_task_size(self, task):
        return self._increments[task]

    def get_dataset(self, indices, source, mode, kshot=None):
        """Build a :class:`DummyDataset` over the classes listed in ``indices``.

        Args:
            indices: mapped class ids to include.
            source: ``"train"`` or ``"test"``.
            mode: ``"train"`` (augmented) or ``"test"`` (deterministic).
            kshot: if an int, incremental classes are subsampled to k images.
        """
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx + 1, kshot=kshot)
            data.append(class_data)
            targets.append(class_targets)

        return DummyDataset(np.concatenate(data), np.concatenate(targets), trsf, self.use_path)

    def _setup_data(self, dataset_name, shuffle, seed):
        idata = get_idata(dataset_name, self.args)
        idata.download_data()

        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path

        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        num_classes = len(np.unique(self._train_targets))
        explicit_order = self.args.get("class_order", None)
        if explicit_order is not None:
            order = [int(v) for v in explicit_order]
            if sorted(order) != list(range(num_classes)):
                raise ValueError(
                    "class_order must be a permutation of 0..{}; got {} entries.".format(
                        num_classes - 1, len(order)
                    )
                )
            logging.info("Using explicit class_order from config.")
        elif shuffle:
            np.random.seed(seed)
            order = np.random.permutation(num_classes).tolist()
        else:
            order = idata.class_order

        self._class_order = order
        # Written back so that models can map class ids to text class names.
        self.args["class_order"] = list(self._class_order)
        logging.info("Class order: %s", self._class_order)

        self._train_targets = _map_new_class_index(self._train_targets, self._class_order)
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def _select(self, x, y, low_range, high_range, kshot=None):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        # Base classes always keep all their data; only incremental classes are k-shot.
        if isinstance(kshot, int) and low_range >= self.args["init_cls"]:
            idxes = list(idxes)
            random.shuffle(idxes)
            idxes = idxes[:kshot]
        return x[idxes], y[idxes]


class DummyDataset(Dataset):
    """Returns ``(index, transformed_image, label)``."""

    def __init__(self, images, labels, trsf, use_path=False):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        return idx, image, self.labels[idx]


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def pil_loader(path):
    with open(path, "rb") as f:
        return Image.open(f).convert("RGB")


class DataIter(object):
    """Infinite iterator over a DataLoader, restarting at the end of each epoch."""

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._iter = iter(self.dataloader)

    def next(self):
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self.dataloader)
            return next(self._iter)
