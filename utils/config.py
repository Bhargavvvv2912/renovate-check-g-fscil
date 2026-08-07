"""Experiment-config loading and lightweight validation.

The public CIFAR, CUB, and ImageNet-R configs are deliberately standalone and
keep the original flat parameter names. ``base_config`` remains supported by
the loader for private ablation configs, but no public experiment depends on it.
"""

import json
import os


SUPPORTED_DATASETS = {"cifar224", "cub", "imagenetr"}
DATASET_SYN_NAMES = {
    "cifar224": "cifar100",
    "cub": "cub",
    "imagenetr": "imagenetr",
}
DATASET_CLASS_COUNTS = {
    "cifar224": 100,
    "cub": 200,
    "imagenetr": 200,
}

REQUIRED_KEYS = {
    "model_name", "backbone_type", "device", "seed", "dataset", "init_cls",
    "increment", "kshot", "vpt_type", "prompt_token_num", "avg_alpha",
    "perturb_var", "EMA_beta", "anchor_lambda", "kl_weight", "TIP_init",
    "tuned_epoch", "fs_epoch", "init_lr", "fs_lr", "batch_size",
    "fs_batch_size", "weight_decay", "min_lr", "optimizer", "shuffle",
    "top_k", "prefix", "model_prefix", "save_dir", "syn_image_path",
    "dataset_syn", "need_syn_imgs", "use_base_syn", "cur_size", "local_bs0",
    "clip_selection_ratio", "clip_diversity_lambda", "mask_scale",
    "mask_lr_mul", "mask_reg_weight", "mask_reg_beta", "rule_mode",
    "rule_tau", "rule_distill_weight", "rule_warmup_epochs", "rule_score_bs",
    "protonet_dynamic_syn", "protonet_syn_ratio", "protonet_syn_per_class",
    "protonet_syn_topk", "protonet_syn_temp", "protonet_clamp_negative_sim",
    "ema_prompt_ignore_syn", "calib_lambda", "calib_m0", "calib_m1",
    "calib_sem_tau", "calib_vis_tau", "calib_fwd_weight", "calib_rev_weight",
    "fs_freeze_fc", "use_old_probes", "class_order",
}

GENERATION_KEYS = {
    "config", "ldm_ckpt", "g_local_train_steps", "com_round_gen",
    "g_local_bs", "n_iter", "synthesis_n_samples", "synthesis_prompt",
    "ddim_steps",
}


def load_config(path, _chain=()):
    """Load a JSON config, recursively resolving its optional base config."""
    path = os.path.abspath(os.path.expanduser(path))
    if path in _chain:
        chain = " -> ".join((*_chain, path))
        raise ValueError("Circular base_config chain: {}".format(chain))
    if not os.path.isfile(path):
        raise FileNotFoundError("Experiment config not found: {}".format(path))

    with open(path, encoding="utf-8") as f:
        child = json.load(f)
    if not isinstance(child, dict):
        raise ValueError("Experiment config must contain a JSON object: {}".format(path))

    parent_ref = child.pop("base_config", None)
    if parent_ref is None:
        return child

    parent_path = os.path.expanduser(str(parent_ref))
    if not os.path.isabs(parent_path):
        parent_path = os.path.join(os.path.dirname(path), parent_path)
    merged = load_config(parent_path, _chain=(*_chain, path))
    merged.update(child)
    return merged


def validate_config(args):
    """Validate contracts that otherwise fail late in a multi-hour run.

    This function intentionally checks structure and value ranges only.  Data
    and checkpoint existence are checked by their owning loaders, where error
    messages can include the exact expected layout.
    """
    missing = sorted(REQUIRED_KEYS - set(args))
    if bool(args.get("need_syn_imgs", False)):
        missing.extend(sorted(GENERATION_KEYS - set(args)))
    if missing:
        raise ValueError("Missing required config keys: {}".format(", ".join(missing)))

    dataset = str(args["dataset"]).lower()
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            "dataset must be one of {}; got {!r}.".format(sorted(SUPPORTED_DATASETS), dataset)
        )
    expected_syn = DATASET_SYN_NAMES[dataset]
    if str(args["dataset_syn"]).lower() != expected_syn:
        raise ValueError(
            "dataset_syn must be {!r} when dataset={!r}; got {!r}.".format(
                expected_syn, dataset, args["dataset_syn"]
            )
        )

    class_count = DATASET_CLASS_COUNTS[dataset]
    try:
        class_order = [int(value) for value in args["class_order"]]
    except (TypeError, ValueError):
        raise ValueError("class_order must be a JSON list of integer class ids.")
    if sorted(class_order) != list(range(class_count)):
        raise ValueError(
            "class_order for {} must be a permutation of 0..{}; got {} entries.".format(
                dataset, class_count - 1, len(class_order)
            )
        )

    if not isinstance(args["seed"], list) or not args["seed"]:
        raise ValueError("seed must be a non-empty JSON list, for example [2024].")
    if not isinstance(args["device"], list) or not args["device"]:
        raise ValueError("device must be a non-empty JSON list, for example [\"0\"].")

    for key in ("init_cls", "increment", "kshot", "batch_size", "fs_batch_size"):
        if int(args[key]) <= 0:
            raise ValueError("{} must be positive; got {}.".format(key, args[key]))
    prompt_tokens = int(args["prompt_token_num"])
    if prompt_tokens <= 0 or prompt_tokens % 2:
        raise ValueError("prompt_token_num must be a positive even integer.")
    if int(args["top_k"]) <= 0 or int(args["top_k"]) > int(args["init_cls"]):
        raise ValueError("top_k must be in [1, init_cls].")

    for key in ("clip_selection_ratio", "protonet_syn_ratio"):
        value = float(args[key])
        if not 0.0 <= value <= 1.0:
            raise ValueError("{} must be in [0, 1]; got {}.".format(key, value))

    if str(args["rule_mode"]).lower() not in {"2d", "3d"}:
        raise ValueError("rule_mode must be either '2d' or '3d'.")
    prompt_mode = str(args.get("syn_prompt_mode", "default")).lower()
    if prompt_mode not in {"default", "prefix"}:
        raise ValueError("syn_prompt_mode must be either 'default' or 'prefix'.")
    if prompt_mode == "prefix" and dataset != "imagenetr":
        raise ValueError("syn_prompt_mode='prefix' is only supported for ImageNet-R.")

    if bool(args.get("need_syn_imgs", False)):
        if "*" not in str(args["synthesis_prompt"]):
            raise ValueError("synthesis_prompt must contain the LDM placeholder token '*'.")
        for key in ("g_local_train_steps", "com_round_gen", "g_local_bs", "n_iter",
                    "synthesis_n_samples", "ddim_steps"):
            if int(args[key]) <= 0:
                raise ValueError("{} must be positive when need_syn_imgs=true.".format(key))

    return args
