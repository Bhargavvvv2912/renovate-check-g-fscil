"""Smoke tests for the data / curation plumbing.

These run on synthetic images in a temp dir and never touch the ViT, CLIP or the
latent diffusion model, so they finish in seconds on CPU:

    python -m pytest tests/test_data_pipeline.py       # or simply run this file
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.asp import (  # noqa: E402
    AugmentedPerClassConcatDataset, Learner, TaskSynImageDataset, dataset_stats,
    syn_train_transform,
)
from trainer import _normalise_devices, harmonic_accuracy  # noqa: E402
from utils.class_names import get_class_names  # noqa: E402
from utils.data_manager import DataManager  # noqa: E402
from utils.gen_data import GenDataset  # noqa: E402
from utils.indexed_concat import build_indexed_concat  # noqa: E402
from utils.mask_bank import MaskBank  # noqa: E402
from utils.toolkit import accuracy  # noqa: E402

N_CLASSES = 20
PER_CLASS_TRAIN = 6
PER_CLASS_TEST = 3


def _write_image(path, seed):
    rng = np.random.RandomState(seed)
    Image.fromarray(rng.randint(0, 255, (40, 40, 3), dtype=np.uint8)).save(path)


def make_fake_imagenetr(root):
    """Build ``<root>/imagenet-r/{train,test}/<wnid>/*.jpg``."""
    dataset_root = os.path.join(root, "imagenet-r")
    os.makedirs(dataset_root, exist_ok=True)
    with open(os.path.join(dataset_root, "classnames.txt"), "w", encoding="utf-8") as f:
        for c in range(N_CLASSES):
            f.write("n{:08d} debug_class_{}\n".format(c, c))

    for split, n in (("train", PER_CLASS_TRAIN), ("test", PER_CLASS_TEST)):
        for c in range(N_CLASSES):
            d = os.path.join(root, "imagenet-r", split, "n{:08d}".format(c))
            os.makedirs(d, exist_ok=True)
            for i in range(n):
                prefix = "art" if i % 2 == 0 else "origami"
                _write_image(os.path.join(d, "{}_{}.jpg".format(prefix, i)), seed=c * 100 + i)


def make_fake_syn(root, task_id, class_ids, n_per_class):
    """Build ``<root>/task_<t>/<class_id>/<class_id>-<i>.jpg``."""
    paths = []
    for c in class_ids:
        d = os.path.join(root, "task_{}".format(task_id), str(c))
        os.makedirs(d, exist_ok=True)
        for i in range(n_per_class):
            p = os.path.join(d, "{}-{}.jpg".format(c, i))
            _write_image(p, seed=1000 + c * 50 + i)
            paths.append(p)
    return paths


def test_class_names_alignment():
    assert len(get_class_names("cifar224")) == 100
    assert len(get_class_names("cub")) == 200
    inr = get_class_names("imagenetr")
    assert len(inr) == 200 and len(set(inr)) == 200
    # sorted-wnid order: n01443537 -> goldfish is the first ImageNet-R class
    assert inr[0] == "goldfish"
    for name in ("imagenetr", "cub", "cifar224"):
        mean, std, size = dataset_stats(name)
        assert len(mean) == 3 and len(std) == 3 and size == 224
        assert syn_train_transform("imagenetr") is not None


def test_data_manager_kshot(tmp_root):
    make_fake_imagenetr(tmp_root)
    args = {
        "dataset": "imagenetr", "data_root": tmp_root, "model_name": "asp",
        "init_cls": 10, "increment": 10,
    }
    dm = DataManager("imagenetr", False, 0, 10, 10, args)
    # the fake set has fewer classes than real ImageNet-R
    dm._class_order = list(range(N_CLASSES))
    dm._increments = [10, 10]

    assert dm.nb_tasks == 2 and dm.nb_classes == N_CLASSES
    assert dm.use_path is True
    assert args["class_order"][:5] == [0, 1, 2, 3, 4]
    assert args["class_names"][0] == "debug class 0"

    base = dm.get_dataset(np.arange(0, 10), source="train", mode="train", kshot=5)
    assert len(base) == 10 * PER_CLASS_TRAIN, "base classes must keep all their data"

    novel = dm.get_dataset(np.arange(10, 20), source="train", mode="train", kshot=5)
    assert len(novel) == 10 * 5, "incremental classes must be capped at k shots"

    test = dm.get_dataset(np.arange(0, 20), source="test", mode="test")
    assert len(test) == N_CLASSES * PER_CLASS_TEST

    idx, img, lab = novel[0]
    assert idx == 0 and img.shape == (3, 224, 224) and 10 <= int(lab) < 20
    return dm, novel


def test_gen_dataset_paths(dm, novel):
    """Textual inversion must work off file paths (CUB / ImageNet-R)."""
    ds = GenDataset(
        images=novel.images, class_ids=novel.labels,
        use_path=dm.use_path, min_class_id=10, repeats=3,
        class_names=dm.args["class_names"], dataset_name="imagenetr",
        syn_prompt_mode="prefix",
    )
    assert len(ds) == len(novel) * 3
    item = ds[0]
    assert item["image"].shape == (256, 256, 3)
    assert -1.0 <= float(item["image"].min()) and float(item["image"].max()) <= 1.0
    assert item["rel_class_id"] == item["abs_class_id"] - 10
    assert "*" in item["caption"]
    assert "debug class" in item["caption"]
    assert "artwork" in item["caption"] or "origami" in item["caption"]

    # decoded-array datasets (CIFAR) go down the other branch
    arr = np.random.RandomState(0).randint(0, 255, (4, 32, 32, 3), dtype=np.uint8)
    ds2 = GenDataset(images=arr, class_ids=np.arange(4), use_path=False, min_class_id=0, repeats=1)
    assert ds2[3]["image"].shape == (256, 256, 3)


def test_syn_dataset_and_mixing(tmp_root, real_ds):
    syn_root = os.path.join(tmp_root, "syn")
    paths = make_fake_syn(syn_root, task_id=0, class_ids=range(10, 20), n_per_class=4)
    assert len(paths) == 40

    # cur_size caps the per-class contribution
    ds = TaskSynImageDataset(paths, size_per_cls=3, transform=syn_train_transform("imagenetr"))
    assert len(ds) == 10 * 3
    idx, img, lab = ds[0]
    assert img.shape == (3, 224, 224) and 10 <= lab < 20

    mixed, total = build_indexed_concat([real_ds, ds])
    assert total == len(real_ds) + len(ds) == len(mixed)
    seen = [mixed[i][0] for i in range(len(mixed))]
    assert seen == list(range(total)), "global indices must be contiguous and unique"

    # the real/synthetic split is what drives syn_flags in the base step
    real_len = len(real_ds)
    gidx = torch.arange(total)
    assert int((gidx < real_len).sum()) == real_len
    return total, real_len


def test_mask_bank(total):
    bank = MaskBank(num_samples=total, scale=65.0, init_val=0.01, device=torch.device("cpu"))
    gidx = torch.arange(min(8, total))
    x = bank.soft(gidx)
    assert x.shape == gidx.shape and bool(((x > 0) & (x < 1)).all())

    x_st = bank.hard_ste(x)
    assert set(x_st.detach().unique().tolist()) <= {0.0, 1.0}
    x_st.sum().backward()
    assert bank.w.grad is not None and float(bank.w.grad.abs().sum()) > 0, "STE must pass gradients"


def test_augmented_dataset(real_ds):
    labels = sorted(set(int(v) for v in real_ds.labels))[:3]
    extras = {lab: [torch.rand(3, 40, 40) for _ in range(2)] for lab in labels}
    scores = {lab: [0.7, 0.3] for lab in labels}

    aug = AugmentedPerClassConcatDataset(
        base_ds=real_ds, extras_by_label=extras,
        extra_transform=lambda t: t, extra_scores_by_label=scores,
    )
    assert len(aug) == len(real_ds) + 2 * len(labels)
    assert len(aug.is_syn) == len(aug) == len(aug.sample_weight) == len(aug.labels)
    assert sum(aug.is_syn) == 2 * len(labels)

    # every returned index must line up with is_syn / sample_weight
    for i in range(len(aug)):
        new_idx, _, lab = aug[i]
        assert new_idx == i and lab == aug.labels[i]
    for lab in labels:
        pos = [i for i in range(len(aug)) if aug.is_syn[i] and aug.labels[i] == lab]
        assert [aug.sample_weight[i] for i in pos] == [0.7, 0.3]
        # extras are spliced in right after the class' last real sample
        real_pos = [i for i in range(len(aug)) if not aug.is_syn[i] and aug.labels[i] == lab]
        assert min(pos) == max(real_pos) + 1


def test_metrics():
    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    y_pred = np.array([0, 1, 1, 1, 2, 2, 3, 0])
    acc = accuracy(y_pred, y_true, nb_old=2, increment=2)
    assert acc["total"] == 75.0
    assert acc["00-01"] == 75.0 and acc["02-03"] == 75.0
    assert acc["old"] == 75.0 and acc["new"] == 75.0

    hacc, old, new = harmonic_accuracy({"00-04": 80.0, "05-09": 40.0, "total": 60.0}, init_cls=5)
    assert (old, new) == (80.0, 40.0)
    assert abs(hacc - 2 * 80 * 40 / 120) < 1e-6
    assert harmonic_accuracy({"00-04": 0.0}, init_cls=5) == (0.0, 0.0, 0.0)


def test_cpu_device_mapping():
    assert _normalise_devices(["cpu"]) == [torch.device("cpu")]
    assert _normalise_devices(["-1"]) == [torch.device("cpu")]
    try:
        _normalise_devices(["cpu", "0"])
    except ValueError:
        pass
    else:
        raise AssertionError("mixed CPU/CUDA config must fail")


def _bare_learner(**attrs):
    """A Learner with no backbone, for testing the pure-python helpers."""
    obj = Learner.__new__(Learner)
    obj.args = attrs.pop("args", {})
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


def test_learner_class_name_mapping():
    # class_order is a permutation, so name lookup must go through it
    order = list(range(200))
    order[0], order[1] = order[1], order[0]
    learner = _bare_learner(args={"dataset": "imagenetr", "class_order": order})
    names = learner._get_class_names()
    raw = get_class_names("imagenetr")
    assert names[0] == raw[1] and names[1] == raw[0]
    assert learner._class_name_for(0) == raw[1]

    # underscores are stripped for prompt readability
    learner = _bare_learner(args={"dataset": "cifar224", "class_order": None})
    assert learner._class_name_for(1) == "aquarium fish"

    # unknown datasets must fail loudly instead of defaulting to CIFAR
    for bad in ({"dataset": "cars196"},):
        try:
            _bare_learner(args=bad)._get_class_names()
        except ValueError:
            pass
        else:
            raise AssertionError("unknown dataset should raise")


def test_learner_syn_path_helpers(tmp_root):
    syn_root = os.path.join(tmp_root, "syn2")
    make_fake_syn(syn_root, task_id=0, class_ids=[3], n_per_class=12)
    make_fake_syn(syn_root, task_id=1, class_ids=[3, 4], n_per_class=2)

    learner = _bare_learner(
        args={"dataset": "imagenetr"}, syn_imgs_dir=syn_root,
        _syn_path_index=None, _cur_task=1,
    )
    # numeric, not lexical, ordering: 2 must come before 10
    cur = learner._get_current_task_syn_paths_for_class(3)
    assert [os.path.basename(p) for p in cur] == ["3-0.jpg", "3-1.jpg"]

    all_tasks = learner._get_syn_paths_for_class(3)
    assert len(all_tasks) == 14
    names = [os.path.basename(p) for p in all_tasks if "task_0" in p]
    assert names[:3] == ["3-0.jpg", "3-1.jpg", "3-2.jpg"]
    assert names[-1] == "3-11.jpg", "sorting must be numeric, got %s" % names[-1]

    assert learner._get_current_task_syn_paths_for_class(999) == []


def test_learner_normalization():
    learner = _bare_learner(args={"dataset": "imagenetr"})
    x = torch.rand(2, 3, 40, 40)
    out = learner._normalize_tensor_batch_for_net(x)
    assert out.shape == (2, 3, 224, 224)

    mean, std, _ = dataset_stats("imagenetr")
    got = float(learner._normalize_tensor_batch_for_net(torch.ones(1, 3, 224, 224))[0, 0, 0, 0])
    assert abs(got - (1.0 - mean[0]) / std[0]) < 1e-5, "wrong normalisation statistics"

    single = learner._normalize_tensor_batch_for_net(torch.rand(3, 224, 224))
    assert single.shape == (1, 3, 224, 224), "3-D input must be promoted to a batch"

    img = Image.fromarray(np.zeros((40, 40, 3), dtype=np.uint8))
    assert learner._net_eval_transform_pil()(img).shape == (3, 224, 224)
    assert learner._protonet_extra_transform_pil()(img).shape == (3, 224, 224)


def main():
    tmp_root = tempfile.mkdtemp(prefix="asp_smoke_")
    try:
        test_class_names_alignment()
        dm, novel = test_data_manager_kshot(tmp_root)
        test_gen_dataset_paths(dm, novel)
        total, _ = test_syn_dataset_and_mixing(tmp_root, novel)
        test_mask_bank(total)
        test_augmented_dataset(novel)
        test_metrics()
        test_cpu_device_mapping()
        test_learner_class_name_mapping()
        test_learner_syn_path_helpers(tmp_root)
        test_learner_normalization()
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    print("all smoke tests passed")


if __name__ == "__main__":
    main()
