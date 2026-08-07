"""Dependency-free tests for standalone configs and prompt routing."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config, validate_config  # noqa: E402
from utils.prompts import (  # noqa: E402
    clip_templates_for_dataset,
    imagenetr_prefix_prompt,
    imagenetr_prompts_for_paths,
    normalize_imagenetr_prefix,
)
from tools.check_imagenetr import check as check_imagenetr  # noqa: E402


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_public_configs():
    expected = {
        "cifar.json": ("cifar224", 5, 5),
        "cub.json": ("cub", 10, 10),
        "inr.json": ("imagenetr", 10, 10),
    }
    configs = {}
    for filename, protocol in expected.items():
        config = load_config(os.path.join(ROOT, "exps", filename))
        validate_config(config)
        assert (config["dataset"], config["init_cls"], config["increment"]) == protocol
        assert "base_config" not in config
        configs[filename] = config

    # Every public JSON is self-contained. ImageNet-R only adds its two
    # dataset-specific path parameters to the same complete parameter surface.
    assert set(configs["cifar.json"]) == set(configs["cub.json"])
    assert set(configs["inr.json"]) - {
        "imagenetr_root", "imagenetr_classnames_txt"
    } == set(configs["cifar.json"])
    assert not os.path.exists(os.path.join(ROOT, "exps", "_base.json"))

    inr = load_config(os.path.join(ROOT, "exps", "inr.json"))
    assert inr["syn_prompt_mode"] == "prefix"
    assert inr["imagenetr_root"].endswith("imagenet-r")


def test_imagenetr_prompt_routing():
    assert normalize_imagenetr_prefix("art_3.jpg") == "artwork"
    assert normalize_imagenetr_prefix("plastic_object_12.png") == "plastic"
    assert imagenetr_prefix_prompt("origami_4.jpg", "goldfish") == "an origami goldfish *"

    prompts = imagenetr_prompts_for_paths(
        ["art_1.jpg", "art_2.jpg", "sketch_1.jpg"], "goldfish"
    )
    assert prompts.count("an artwork of a goldfish *") == 2
    assert prompts.count("a sketch of a goldfish *") == 1
    assert len(clip_templates_for_dataset("imagenetr")) > 1
    assert len(clip_templates_for_dataset("cub")) == 1


def test_imagenetr_layout_checker():
    with tempfile.TemporaryDirectory(prefix="inr_check_") as root:
        for split in ("train", "test"):
            for wnid in ("n00000001", "n00000002"):
                class_dir = os.path.join(root, split, wnid)
                os.makedirs(class_dir)
                with open(os.path.join(class_dir, "art_0.jpg"), "wb") as f:
                    f.write(b"layout-only test")
        with open(os.path.join(root, "classnames.txt"), "w", encoding="utf-8") as f:
            f.write("n00000001 debug one\n")
            f.write("n00000002 debug two\n")
        assert check_imagenetr(root, allow_subset=True) == 0


def main():
    test_public_configs()
    test_imagenetr_prompt_routing()
    test_imagenetr_layout_checker()
    print("config and prompt tests passed")


if __name__ == "__main__":
    main()
