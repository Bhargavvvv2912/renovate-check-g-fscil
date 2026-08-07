#!/usr/bin/env python3
"""Validate an ImageNet-R train/test tree before starting a long run."""

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.class_names import load_imagenetr_classname_map  # noqa: E402
from utils.prompts import normalize_imagenetr_prefix  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _class_dirs(root, split):
    split_root = os.path.join(root, split)
    if not os.path.isdir(split_root):
        raise FileNotFoundError("Missing split directory: {}".format(split_root))
    return sorted(
        name for name in os.listdir(split_root)
        if os.path.isdir(os.path.join(split_root, name))
    )


def _image_paths(root, split, classes):
    for wnid in classes:
        class_root = os.path.join(root, split, wnid)
        for name in sorted(os.listdir(class_root)):
            path = os.path.join(class_root, name)
            if os.path.isfile(path) and os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                yield path


def check(root, classnames_txt=None, allow_subset=False):
    root = os.path.abspath(os.path.expanduser(root))
    train_classes = _class_dirs(root, "train")
    test_classes = _class_dirs(root, "test")
    if train_classes != test_classes:
        train_only = sorted(set(train_classes) - set(test_classes))
        test_only = sorted(set(test_classes) - set(train_classes))
        raise ValueError(
            "train/test WNIDs differ. train-only={} test-only={}".format(
                train_only[:10], test_only[:10]
            )
        )
    if not train_classes:
        raise ValueError("No class folders found under {}.".format(root))
    if len(train_classes) != 200 and not allow_subset:
        raise ValueError(
            "Expected 200 ImageNet-R classes, found {}. Use --allow-subset only for a "
            "deliberate debug subset.".format(len(train_classes))
        )

    train_paths = list(_image_paths(root, "train", train_classes))
    test_paths = list(_image_paths(root, "test", test_classes))
    if not train_paths or not test_paths:
        raise ValueError("Both train and test splits must contain image files.")

    explicit_classname_path = classnames_txt is not None
    classnames_txt = classnames_txt or os.path.join(root, "classnames.txt")
    if explicit_classname_path and not os.path.isfile(classnames_txt):
        raise FileNotFoundError("Class-name file not found: {}".format(classnames_txt))
    name_map = load_imagenetr_classname_map(classnames_txt)
    missing_names = [wnid for wnid in train_classes if wnid not in name_map] if name_map else []

    prefix_counts = Counter(normalize_imagenetr_prefix(path) for path in train_paths)
    unrecognised = int(prefix_counts.pop(None, 0))
    print("ImageNet-R root: {}".format(root))
    print("Classes: {} | train images: {} | test images: {}".format(
        len(train_classes), len(train_paths), len(test_paths)
    ))
    if name_map:
        print("Class names: {} ({} missing WNIDs)".format(classnames_txt, len(missing_names)))
    else:
        print("Class names: no classnames.txt; the bundled 200-name fallback will be used")
    print("Recognised rendition prefixes: {}/{} | top: {}".format(
        len(train_paths) - unrecognised,
        len(train_paths),
        prefix_counts.most_common(8),
    ))
    if missing_names:
        raise ValueError("classnames.txt is missing WNIDs: {}".format(missing_names[:10]))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="./data/imagenet-r")
    parser.add_argument("--classnames-txt", default=None)
    parser.add_argument("--allow-subset", action="store_true")
    args = parser.parse_args()
    try:
        return check(args.root, args.classnames_txt, args.allow_subset)
    except (FileNotFoundError, ValueError) as exc:
        print("ImageNet-R check failed: {}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
