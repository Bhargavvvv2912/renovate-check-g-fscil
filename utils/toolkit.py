import numpy as np


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def accuracy(y_pred, y_true, nb_old, increment=10):
    """Overall, per-class-group, old and new accuracy (in percent)."""
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {"total": np.around((y_pred == y_true).sum() * 100 / len(y_true), decimals=2)}

    for class_id in range(0, np.max(y_true), increment):
        idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + increment))[0]
        label = "{}-{}".format(
            str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
        )
        all_acc[label] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )

    old_idxes = np.where(y_true < nb_old)[0]
    all_acc["old"] = 0 if len(old_idxes) == 0 else np.around(
        (y_pred[old_idxes] == y_true[old_idxes]).sum() * 100 / len(old_idxes), decimals=2
    )

    new_idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = np.around(
        (y_pred[new_idxes] == y_true[new_idxes]).sum() * 100 / len(new_idxes), decimals=2
    )
    return all_acc


def split_images_labels(imgs):
    """Split ``ImageFolder.imgs`` into parallel path / label arrays."""
    images = [item[0] for item in imgs]
    labels = [item[1] for item in imgs]
    return np.array(images), np.array(labels)
