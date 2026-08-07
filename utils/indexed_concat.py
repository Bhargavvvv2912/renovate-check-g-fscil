"""Concatenate datasets while exposing a contiguous global sample index.

The base-session curation mask stores one parameter per training sample, so
every batch must carry the index of each sample inside the *concatenated*
dataset rather than inside its own sub-dataset.
"""

from torch.utils.data import ConcatDataset, Dataset


class WithGlobalIndex(Dataset):
    """Shift a sub-dataset's indices by ``offset`` and return them as field 0.

    The wrapped dataset may yield either ``(local_idx, img, label)`` or
    ``(img, label)``.
    """

    def __init__(self, base_ds: Dataset, offset: int):
        self.base_ds = base_ds
        self.offset = offset

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, i):
        item = self.base_ds[i]
        if len(item) == 3:
            _, img, label = item
        elif len(item) == 2:
            img, label = item
        else:
            raise ValueError("Dataset item must return 2 or 3 elements, got {}".format(len(item)))
        return self.offset + i, img, label


def build_indexed_concat(datasets):
    """Return ``(concat_dataset, total_length)`` with globally unique indices."""
    wrapped, offset = [], 0
    for ds in datasets:
        wrapped.append(WithGlobalIndex(ds, offset))
        offset += len(ds)
    return ConcatDataset(wrapped), offset
