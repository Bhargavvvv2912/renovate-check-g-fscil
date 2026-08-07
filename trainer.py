"""Task loop: runs every seed, drives the learner and records per-task metrics."""

import copy
import json
import logging
import os
import random
import time

import numpy as np
import torch

from utils.data_manager import DataManager
from utils.toolkit import count_parameters

MODELS = {"asp": ("models.asp", "Learner")}


def get_model(model_name, args):
    key = str(model_name).lower()
    if key not in MODELS:
        raise ValueError("Unknown model_name '{}'. Supported: {}.".format(model_name, sorted(MODELS)))
    module_name, cls_name = MODELS[key]
    module = __import__(module_name, fromlist=[cls_name])
    return getattr(module, cls_name)(args)


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    raw_devices = copy.deepcopy(args["device"])
    for seed in seed_list:
        run_args = copy.deepcopy(args)
        run_args["seed"] = seed
        run_args["device"] = copy.deepcopy(raw_devices)
        _train(run_args)


def _train(args):
    init_cls = args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}_{}".format(
        args["model_name"], args["dataset"], init_cls, args["increment"], args["kshot"]
    )
    saved_path = "saved_model/{}/{}/{}_{}".format(
        args["model_name"], args["dataset"], init_cls, args["increment"]
    )
    os.makedirs(logs_name, exist_ok=True)
    os.makedirs(saved_path, exist_ok=True)

    args["base_model_path"] = os.path.join(
        saved_path,
        "{}_{}_{}_{}.pth".format(
            args["model_prefix"], args["tuned_epoch"], args["seed"], args["backbone_type"]
        ),
    )

    _set_random(args["seed"])
    _set_device(args)
    _log_args(args)

    data_manager = DataManager(
        args["dataset"], args["shuffle"], args["seed"],
        args["init_cls"], args["increment"], args,
    )
    args["nb_classes"] = data_manager.nb_classes
    args["nb_tasks"] = data_manager.nb_tasks

    model = get_model(args["model_name"], args)

    top1_curve = []
    for task in range(data_manager.nb_tasks):
        task_token = _timing_start(args)
        logging.info("All params: %d", count_parameters(model._network))
        logging.info("Trainable params: %d", count_parameters(model._network, True))

        train_token = _timing_start(args)
        model.incremental_train(data_manager)
        _timing_end(args, train_token, "trainer_incremental_train", task)

        eval_token = _timing_start(args)
        result = model.eval_task()
        _timing_end(args, eval_token, "trainer_eval_task", task)

        after_token = _timing_start(args)
        model.after_task()
        _timing_end(args, after_token, "trainer_after_task", task)

        cur_top1 = float(result["top1"])
        top1_curve.append(cur_top1)
        avg_top1 = sum(top1_curve) / len(top1_curve)
        hacc, old_acc, new_acc = harmonic_accuracy(result["grouped"], args["init_cls"])

        logging.info("[Task %d] Top1: %.2f", task, cur_top1)
        logging.info("Top1 curve: %s", [round(x, 4) for x in top1_curve])
        logging.info(
            "Average Top1: %.2f | Harmonic: %.2f | Old: %.2f | New: %.2f\n",
            avg_top1, hacc, old_acc, new_acc,
        )

        _append_jsonl(
            args, "perf",
            {
                "task": int(task),
                "seed": int(args["seed"]),
                "dataset": args.get("dataset"),
                "prefix": args.get("prefix"),
                "top1": cur_top1,
                "top1_curve": [float(x) for x in top1_curve],
                "avg_top1": float(avg_top1),
                "hacc": float(hacc),
                "old_acc": float(old_acc),
                "new_acc": float(new_acc),
            },
        )
        _timing_end(
            args, task_token, "trainer_task_loop_total", task,
            top1=cur_top1, avg_top1=float(avg_top1), hacc=float(hacc),
            old_acc=float(old_acc), new_acc=float(new_acc),
        )


def harmonic_accuracy(grouped_acc, init_cls):
    """Harmonic mean of base-class and novel-class accuracy.

    ``grouped_acc`` keys look like ``"00-04"``; a group belongs to the base
    session when its lower bound is smaller than ``init_cls``.
    """
    old_acc_list, new_acc_list = [], []
    for key, value in grouped_acc.items():
        if "-" not in key:
            continue
        lower = int(key.split("-")[0])
        (old_acc_list if lower < init_cls else new_acc_list).append(value)

    old_acc = sum(old_acc_list) / len(old_acc_list) if old_acc_list else 0.0
    new_acc = sum(new_acc_list) / len(new_acc_list) if new_acc_list else 0.0
    if old_acc + new_acc == 0.0:
        return 0.0, old_acc, new_acc
    return 2 * old_acc * new_acc / (old_acc + new_acc), old_acc, new_acc


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #
def _metrics_path(args, kind):
    metrics_dir = os.path.join(args.get("save_dir", "outputs"), "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    return os.path.join(
        metrics_dir, "{}_{}_seed{}.jsonl".format(kind, args.get("prefix", "run"), args["seed"])
    )


def _append_jsonl(args, kind, record):
    with open(_metrics_path(args, kind), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _timing_start(args):
    if not bool(args.get("timing_enabled", False)):
        return None
    _sync_timing_devices(args)
    return {"wall": time.perf_counter(), "cpu": time.process_time()}


def _timing_end(args, token, phase, task, **extra):
    if token is None:
        return
    _sync_timing_devices(args)
    record = {
        "phase": phase,
        "task": int(task),
        "seed": int(args["seed"]),
        "dataset": args.get("dataset"),
        "prefix": args.get("prefix"),
        "wall_sec": float(time.perf_counter() - token["wall"]),
        "cpu_sec": float(time.process_time() - token["cpu"]),
    }
    record.update(extra)
    _append_jsonl(args, "timing", record)
    logging.info("[Timing] %s | task=%d | wall=%.2fs", phase, task, record["wall_sec"])


def _sync_timing_devices(args):
    if not bool(args.get("timing_sync_cuda", True)) or not torch.cuda.is_available():
        return
    for dev in args.get("device", []):
        if isinstance(dev, torch.device) and dev.type == "cuda":
            torch.cuda.synchronize(dev)


def _normalise_devices(device_cfg):
    """Resolve config device ids against the process-visible CUDA devices."""
    if not isinstance(device_cfg, (list, tuple)):
        device_cfg = [device_cfg]

    raw = [str(value).strip().lower() for value in device_cfg]
    if any(value in {"-1", "cpu"} for value in raw):
        if len(raw) != 1:
            raise ValueError("CPU cannot be mixed with CUDA entries in device.")
        return [torch.device("cpu")]

    indices = []
    for value in raw:
        if value.startswith("cuda:"):
            value = value.split(":", 1)[1]
        if not value.isdigit():
            raise ValueError(
                "Invalid device {!r}; use logical CUDA ids such as [\"0\"] or [\"cpu\"].".format(
                    value
                )
            )
        indices.append(int(value))

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA devices {} were requested, but torch.cuda.is_available() is false. "
            "Use device=[\"cpu\"] only for smoke/debug runs.".format(indices)
        )
    visible_count = torch.cuda.device_count()
    invalid = [idx for idx in indices if idx >= visible_count]
    if invalid:
        raise RuntimeError(
            "Invalid logical CUDA id(s) {}: this process sees {} device(s). "
            "When launching with CUDA_VISIBLE_DEVICES=<physical-id>, keep device=[\"0\"] "
            "inside the JSON config.".format(invalid, visible_count)
        )
    return [torch.device("cuda:{}".format(idx)) for idx in indices]


def _set_device(args):
    args["device"] = _normalise_devices(args["device"])
    logging.info("Using devices: %s", args["device"])


def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def _log_args(args):
    for key, value in args.items():
        logging.info("%s: %s", key, value)
