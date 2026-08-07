"""ASP learner: prompt-tuned ViT + diffusion replay with learnable data curation.

Pipeline per task
-----------------
Base session (task 0)
    1. (optional) class inversion + latent-diffusion synthesis of extra images
       for the base classes -- see ``_class_inversion`` / ``_synthesis_imgs``.
    2. Real base images and synthetic base images are concatenated into one
       globally indexed dataset so that every sample owns a slot in ``MaskBank``.
    3. The backbone, the per-sample curation mask and the transferable rule
       scorer are trained jointly. The mask is supervised by CLIP semantic and
       diversity scores plus a budget term; the rule scorer is distilled from
       the converged mask so that curation transfers to unseen classes.
    4. The classifier head is rebuilt from real base prototypes.

Incremental sessions (task > 0)
    1. Synthetic images for the new classes are scored by the *frozen* rule
       scorer; the top-k per class are appended to the k-shot real support set.
    2. Prototypes are rebuilt from that augmented support set, weighted by the
       rule's soft scores (3d mode) or by prototype similarity (2d mode).
    3. Prompts are fine-tuned with a masked cross-entropy plus the bidirectional
       gated calibration loss (CVS-GC), which pushes new classes away from
       semantically/visually confusable old classes and, symmetrically, keeps
       cached old-class probes from being absorbed by the new head.
"""

import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from glob import glob

try:
    import resource
except ImportError:  # pragma: no cover - Windows
    resource = None

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm, trange

from backbone.asp_backbone import SimpleVitNet
from models.base import BaseLearner
from models.clip_selection import (
    CLIP_MODEL_NAME,
    build_clip_text_features,
    compute_clip_scores_memory,
    get_clip_model,
)
from utils.class_names import get_class_names
from utils.data_manager import DataIter
from utils.gen_data import GenDataset
from utils.indexed_concat import build_indexed_concat
from utils.mask_bank import MaskBank
from utils.prompts import (
    clip_templates_for_dataset,
    format_class_synthesis_prompt,
    imagenetr_prompts_for_paths,
)

DEFAULT_NUM_WORKERS = 2

# (mean, std, input resolution) used to normalise synthetic images before they
# are fed to the ViT backbone. Must match the corresponding entry in utils/data.py.
DATASET_STATS = {
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761), 224),
    "cifar224": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761), 224),
    "cub": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 224),
    "cub200": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 224),
    "imagenetr": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 224),
}

# Augmentation applied to synthetic PIL images that join the base-session
# training mix. Keyed by ``dataset_syn``.
SYN_TRAIN_TRANSFORMS = {
    "cifar100": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
    ]),
    "cub": transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]),
    "imagenetr": transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]),
}


# Normalisation used for synthetic prototype candidates. Kept as a separate
# constant because the original code always used CIFAR-100 statistics here,
# regardless of the dataset actually being trained on.
PROTONET_EXTRA_STATS = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))


def dataset_stats(dataset_name):
    """Return ``(mean, std, size)`` for ``dataset_name``."""
    key = str(dataset_name).lower()
    if key not in DATASET_STATS:
        raise ValueError(
            "No normalisation statistics registered for dataset '{}'. "
            "Add an entry to models/asp.py:DATASET_STATS.".format(dataset_name)
        )
    return DATASET_STATS[key]


def syn_train_transform(dataset_syn):
    """Return the augmentation used for synthetic images of ``dataset_syn``."""
    key = str(dataset_syn).lower()
    if key not in SYN_TRAIN_TRANSFORMS:
        raise ValueError(
            "No synthetic-image transform registered for dataset_syn='{}'. "
            "Add an entry to models/asp.py:SYN_TRAIN_TRANSFORMS.".format(dataset_syn)
        )
    return SYN_TRAIN_TRANSFORMS[key]


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = SimpleVitNet(args, True)

        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8

        if args.get("syn_image_path"):
            self.syn_imgs_dir = args["syn_image_path"]
        else:
            self.syn_imgs_dir = os.path.join(args["save_dir"], "syn_imgs")

        # ---- state shared across tasks ----
        self.local_cur_loaders = []
        self.mask_bank = None
        self.sim_scores = None
        self.div_scores = None
        self._syn_path_index = None
        self._cached_syn_gpu = {}
        self._generator = None
        self.gen_data_iter = None

        # ---- base-session mask logging ----
        self._real_len = 0
        self._syn_len = 0
        self._epoch_real_kept = 0.0
        self._epoch_syn_kept = 0.0
        self._epoch_real_total = 0
        self._epoch_syn_total = 0

        # ---- CVS-GC gated calibration ----
        self.calib_lambda = float(args.get("calib_lambda", 1.0))     # overall weight
        self.calib_m0 = float(args.get("calib_m0", 0.1))             # base margin
        self.calib_m1 = float(args.get("calib_m1", 0.0))             # gate-scaled margin bonus
        self.calib_sem_tau = float(args.get("calib_sem_tau", 0.5))   # CLIP-similarity gate threshold
        self.calib_vis_tau = float(args.get("calib_vis_tau", 0.05))  # visual-confusion gate margin
        self.gate_matrix = None                                      # [num_old, num_new] for this task
        self._clip_model = None
        self.use_old_probes = bool(args.get("use_old_probes", True))

        # ---- transferable rule scorer (learned in the base session only) ----
        self.rule_mode = str(args.get("rule_mode", "3d")).lower()
        if self.rule_mode not in ("3d", "2d"):
            raise ValueError("rule_mode must be '3d' or '2d', got {!r}".format(self.rule_mode))
        self.rule_scorer = nn.Linear(3 if self.rule_mode == "3d" else 2, 1, bias=True)
        nn.init.zeros_(self.rule_scorer.weight)   # start from a neutral score
        nn.init.zeros_(self.rule_scorer.bias)
        self.rule_tau = float(args.get("rule_tau", 5.0))
        self.rule_distill_weight = float(args.get("rule_distill_weight", 1.0))
        self.rule_warmup_epochs = int(args.get("rule_warmup_epochs", 1))

        # ---- profiling ----
        self._eval_each_epoch = bool(args.get("eval_each_epoch", False))
        self._timing_enabled = bool(args.get("timing_enabled", False))
        self._timing_sync_cuda = bool(args.get("timing_sync_cuda", True))
        self._timing_jsonl = os.path.join(
            args["save_dir"], "metrics",
            "timing_{}_seed{}.jsonl".format(args.get("prefix", "run"), args.get("seed", "na")),
        )

    # ------------------------------------------------------------------ #
    # profiling helpers
    # ------------------------------------------------------------------ #
    def _cuda_timing_device(self):
        if not torch.cuda.is_available():
            return None
        dev = getattr(self, "_device", None)
        if isinstance(dev, torch.device) and dev.type == "cuda":
            return dev
        return None

    def _sync_for_timing(self):
        dev = self._cuda_timing_device()
        if self._timing_sync_cuda and dev is not None:
            torch.cuda.synchronize(dev)

    @staticmethod
    def _cpu_seconds():
        if resource is None:
            return time.process_time()
        usage = resource.getrusage(resource.RUSAGE_SELF)
        cpu_sec = usage.ru_utime + usage.ru_stime
        try:
            child = resource.getrusage(resource.RUSAGE_CHILDREN)
            cpu_sec += child.ru_utime + child.ru_stime
        except Exception:
            pass
        return cpu_sec

    @staticmethod
    def _rss_mb():
        if resource is None:
            return None
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # Linux reports KB, macOS reports bytes.
        denom = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
        return float(usage.ru_maxrss) / denom

    def _timing_begin(self):
        if not self._timing_enabled:
            return None
        self._sync_for_timing()
        dev = self._cuda_timing_device()
        if dev is not None:
            try:
                torch.cuda.reset_peak_memory_stats(dev)
            except Exception:
                pass
        return {"wall": time.perf_counter(), "cpu": self._cpu_seconds()}

    def _timing_end(self, token, phase, **extra):
        if token is None:
            return
        self._sync_for_timing()
        dev = self._cuda_timing_device()
        record = {
            "phase": phase,
            "task": int(self._cur_task),
            "known_classes": int(self._known_classes),
            "total_classes": int(self._total_classes),
            "seed": int(self.args.get("seed", -1)),
            "dataset": self.args.get("dataset"),
            "prefix": self.args.get("prefix"),
            "wall_sec": float(time.perf_counter() - token["wall"]),
            "cpu_sec": float(self._cpu_seconds() - token["cpu"]),
            "rss_max_mb": self._rss_mb(),
        }
        if dev is not None:
            record["cuda_peak_alloc_mb"] = float(torch.cuda.max_memory_allocated(dev)) / (1024.0 ** 2)
            record["cuda_peak_reserved_mb"] = float(torch.cuda.max_memory_reserved(dev)) / (1024.0 ** 2)
        record.update(extra)

        os.makedirs(os.path.dirname(self._timing_jsonl), exist_ok=True)
        with open(self._timing_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logging.info(
            "[Timing] %s | task=%s | wall=%.2fs | cpu=%.2fs",
            phase, self._cur_task, record["wall_sec"], record["cpu_sec"],
        )

    @contextmanager
    def _timed(self, phase, **extra):
        token = self._timing_begin()
        try:
            yield
        finally:
            self._timing_end(token, phase, **extra)

    # ------------------------------------------------------------------ #
    # small utilities
    # ------------------------------------------------------------------ #
    def after_task(self):
        self._known_classes = self._total_classes

    def _worker_count(self, key, default):
        return int(self.args.get(key, default))

    def _get_class_names(self):
        """Class names indexed by the *mapped* (post class-order) labels."""
        base_names = self.args.get("class_names")
        if base_names is None:
            base_names = get_class_names(self.args.get("dataset", "cifar224"))
        base_names = list(base_names)
        class_order = self.args.get("class_order", None)
        if class_order is None:
            return base_names
        if len(class_order) != len(base_names):
            logging.warning(
                "Ignoring class_order for class names: %d ids for %d names.",
                len(class_order), len(base_names),
            )
            return base_names
        return [base_names[int(orig_id)] for orig_id in class_order]

    def _class_name_for(self, class_id):
        names = self._get_class_names()
        class_id = int(class_id)
        name = names[class_id] if class_id < len(names) else str(class_id)
        return str(name).replace("_", " ")

    def _clip_templates(self):
        return clip_templates_for_dataset(self.args.get("dataset", "cifar224"))

    def _ds_mean_std(self):
        return dataset_stats(self.args.get("dataset", "cifar224"))

    def _net_eval_transform_pil(self):
        """PIL -> normalised tensor, matching the backbone's test-time pipeline."""
        mean, std, size = self._ds_mean_std()
        return transforms.Compose([
            transforms.Resize((size, size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def _normalize_tensor_batch_for_net(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise a batch of ``[0, 1]`` tensors for the backbone."""
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.float()

        mean, std, size = self._ds_mean_std()
        if x.shape[-2] != size or x.shape[-1] != size:
            x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)

        mean_t = torch.tensor(mean, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        std_t = torch.tensor(std, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        return (x - mean_t) / std_t

    # ------------------------------------------------------------------ #
    # synthetic image bookkeeping
    # ------------------------------------------------------------------ #
    @staticmethod
    def _syn_path_sort_key(path):
        """Sort ``<class>-<counter>.jpg`` numerically rather than lexically."""
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            return int(stem.split("-")[-1])
        except Exception:
            return stem

    def _get_syn_paths_for_class(self, class_id):
        """All synthetic images ever generated for ``class_id`` (any task)."""
        if self._syn_path_index is None:
            self._syn_path_index = defaultdict(list)
            for path in glob(os.path.join(self.syn_imgs_dir, "task_*", "*", "*.jpg")):
                try:
                    cls = int(os.path.basename(os.path.dirname(path)))
                except ValueError:
                    continue
                self._syn_path_index[cls].append(path)
            for paths in self._syn_path_index.values():
                paths.sort(key=self._syn_path_sort_key)
        return list(self._syn_path_index.get(int(class_id), []))

    def _get_current_task_syn_paths_for_class(self, class_id):
        """Synthetic images generated for ``class_id`` during the current task."""
        lab_dir = os.path.join(self.syn_imgs_dir, "task_{}".format(self._cur_task), str(int(class_id)))
        if not os.path.isdir(lab_dir):
            return []
        return sorted(glob(os.path.join(lab_dir, "*.jpg")), key=self._syn_path_sort_key)

    # ------------------------------------------------------------------ #
    # prototypes and the transferable rule
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _compute_real_class_prototypes(self, loader: DataLoader):
        """Mean backbone feature per class over ``loader`` (real support set)."""
        self._network.eval()

        feat_list, lab_list = [], []
        for _, data, label in loader:
            data = data.to(self._device)
            out = self._network(data, self.args.get("perturb_var", None))
            feat_list.append(out["features"].detach().cpu())
            lab_list.append(label.detach().cpu())

        if not feat_list:
            return {}

        feats = torch.cat(feat_list, dim=0)
        labs = torch.cat(lab_list, dim=0)
        return {int(c): feats[labs == int(c)].mean(dim=0) for c in torch.unique(labs)}

    def _build_rule_input_from_batch(self, gidx, targets, feat_actual, syn_flags, real_proto_dict=None):
        """Build the rule input ``z`` for the synthetic samples of one batch.

        3d: ``[clip_sim, proto_sim, diversity]``; 2d: ``[clip_sim, diversity]``.
        """
        syn_mask = syn_flags.bool()
        if syn_mask.sum().item() == 0:
            return None, None

        sim_b = self.sim_scores[gidx].float()[syn_mask]
        div_b = self.div_scores[gidx].float()[syn_mask]

        if self.rule_mode == "2d":
            return torch.stack([sim_b, div_b], dim=1), syn_mask

        feat_syn = feat_actual[syn_mask].detach().cpu()
        lab_syn = targets.detach().cpu()[syn_mask.detach().cpu()]

        proto_vals = []
        for f_i, c in zip(feat_syn, lab_syn):
            mu = real_proto_dict.get(int(c), None) if real_proto_dict else None
            if mu is None:
                proto_vals.append(0.0)
            else:
                f_n = F.normalize(f_i.unsqueeze(0), dim=1).squeeze(0)
                mu_n = F.normalize(mu.unsqueeze(0), dim=1).squeeze(0)
                proto_vals.append(float(torch.dot(f_n, mu_n).item()))

        proto_b = torch.tensor(proto_vals, dtype=torch.float32, device=self._device)
        return torch.stack([sim_b, proto_b, div_b], dim=1), syn_mask

    @torch.no_grad()
    def _score_rule_candidates(self, cand_tensors, cand_labels, real_proto_dict=None):
        """Score candidate synthetic images with the frozen rule scorer.

        Returns ``{"q": raw score, "m": sigmoid(rule_tau * q) soft weight}``.
        """
        if len(cand_tensors) == 0:
            return {"q": torch.empty(0), "m": torch.empty(0)}

        self.rule_scorer.eval()
        self._network.eval()

        # 1) CLIP semantic + diversity scores
        uniq = sorted(set(int(c) for c in cand_labels))
        dataset_class_names = self._get_class_names()
        class_names = [dataset_class_names[c] for c in uniq]

        clip_token = self._timing_begin()
        sim_scores, div_scores = compute_clip_scores_memory(
            cand_tensors, cand_labels, class_names, device=self._device,
            templates=self._clip_templates(),
        )
        self._timing_end(
            clip_token, "rule_candidate_clip_score",
            candidate_count=int(len(cand_tensors)), candidate_classes=int(len(uniq)),
        )
        sim_scores = sim_scores.float().cpu()
        div_scores = div_scores.float().cpu()

        # 2) assemble z
        if self.rule_mode == "2d":
            z = torch.stack([sim_scores, div_scores], dim=1).to(self._device)
        else:
            proto_token = self._timing_begin()
            bs = int(self.args.get("rule_score_bs", 64))
            feat_list = []
            for start in range(0, len(cand_tensors), bs):
                end = min(start + bs, len(cand_tensors))
                batch = torch.stack([cand_tensors[k].float() for k in range(start, end)], dim=0)
                batch = self._normalize_tensor_batch_for_net(batch.to(self._device))
                out = self._network(batch, self.args.get("perturb_var", None))
                feat_list.append(out["features"].detach().cpu())
            feats = torch.cat(feat_list, dim=0)

            proto_scores = []
            for i, c in enumerate(cand_labels):
                mu = real_proto_dict.get(int(c), None) if real_proto_dict else None
                if mu is None:
                    proto_scores.append(0.0)
                else:
                    f_n = F.normalize(feats[i].unsqueeze(0), dim=1).squeeze(0)
                    mu_n = F.normalize(mu.unsqueeze(0), dim=1).squeeze(0)
                    proto_scores.append(float(torch.dot(f_n, mu_n).item()))

            z = torch.stack(
                [sim_scores, torch.tensor(proto_scores, dtype=torch.float32), div_scores], dim=1
            ).to(self._device)
            self._timing_end(
                proto_token, "rule_candidate_proto_score",
                candidate_count=int(len(cand_tensors)), rule_score_bs=int(bs),
            )

        # 3) rule score
        rule_token = self._timing_begin()
        q = self.rule_scorer(z).squeeze(1)
        m = torch.sigmoid(self.rule_tau * q)
        self._timing_end(
            rule_token, "rule_candidate_mlp_score",
            candidate_count=int(len(cand_tensors)), rule_mode=str(self.rule_mode),
        )
        return {"q": q.detach().cpu(), "m": m.detach().cpu()}

    @torch.no_grad()
    def replace_fc(self, trainloader, model, args=None):
        """Rebuild the cosine classifier from class prototypes.

        3d mode fuses the selected synthetic embeddings with the learned rule
        weights. 2d mode falls back to prototype-similarity top-k with softmax
        weighting, because in that mode the rule only gates candidate selection.
        """
        model = model.eval()

        emb_list, lab_list, idx_list = [], [], []
        for idx, data, label in trainloader:
            data = data.to(self._device)
            feat = model(data)["features"]
            emb_list.append(feat.detach().cpu())
            lab_list.append(label.detach().cpu())
            idx_list.append(idx.detach().cpu())

        emb = torch.cat(emb_list, dim=0)
        lab = torch.cat(lab_list, dim=0)
        idx = torch.cat(idx_list, dim=0)

        ds = trainloader.dataset
        if hasattr(ds, "is_syn"):
            is_syn = torch.tensor(ds.is_syn, dtype=torch.bool)[idx].clone()
        else:
            is_syn = torch.zeros_like(lab, dtype=torch.bool)

        if hasattr(ds, "sample_weight"):
            sample_weight = torch.tensor(ds.sample_weight, dtype=torch.float32)[idx].clone()
        else:
            sample_weight = torch.ones(len(idx), dtype=torch.float32)

        use_dyn = bool(self.args.get("protonet_dynamic_syn", True))
        topk = int(self.args.get("protonet_syn_topk", 5))
        syn_ratio = float(self.args.get("protonet_syn_ratio", 0.1))
        temp = float(self.args.get("protonet_syn_temp", 0.07))
        clamp_neg = bool(self.args.get("protonet_clamp_negative_sim", True))

        for c in torch.unique(lab).tolist():
            m_c = lab == int(c)
            emb_c = emb[m_c]
            if emb_c.numel() == 0:
                continue

            syn_c = is_syn[m_c]
            real_c = ~syn_c
            w_c = sample_weight[m_c]

            mu_real = emb_c[real_c].mean(dim=0) if real_c.any() else emb_c.mean(dim=0)
            proto = mu_real

            if use_dyn and syn_ratio > 0 and syn_c.any():
                emb_syn = emb_c[syn_c]

                if self.rule_mode == "3d":
                    w_syn = w_c[syn_c].float()
                    if w_syn.sum().item() > 1e-8:
                        w_syn = w_syn / w_syn.sum()
                        syn_term = (w_syn.unsqueeze(1) * emb_syn).sum(dim=0)
                        proto = (mu_real + syn_ratio * syn_term) / (1.0 + syn_ratio)
                else:
                    sims = F.normalize(emb_syn, dim=1) @ F.normalize(mu_real, dim=0)
                    if clamp_neg:
                        sims = torch.clamp(sims, min=0.0)
                    k = min(topk, emb_syn.size(0))
                    if k > 0:
                        vals, inds = torch.topk(sims, k=k, largest=True)
                        w = F.softmax(vals / max(temp, 1e-6), dim=0)
                        syn_term = (w.unsqueeze(1) * emb_syn[inds]).sum(dim=0)
                        proto = (mu_real + syn_ratio * syn_term) / (1.0 + syn_ratio)

            self._network.fc.weight.data[int(c)] = proto.to(self._network.fc.weight.data.device)

        return model

    # ------------------------------------------------------------------ #
    # CVS-GC gating
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _prepare_calibration_gates(self):
        """Fuse a semantic prior ``S`` and a visual-confusion matrix ``Q`` into
        the gate matrix ``G = sigmoid(a(S - tau_s)) * sigmoid(b(Q - tau_v))``.

        A pair (old class, new class) is only calibrated when the two classes
        are *both* semantically close (CLIP text similarity) and visually
        confusable (the new class' synthetic images leak probability mass onto
        the old class under the current model).
        """
        if self._cur_task == 0:
            return

        device = self._device
        old_classes = list(range(self._known_classes))
        new_classes = list(range(self._known_classes, self._total_classes))

        if self._clip_model is None:
            # Reuse the same ViT-B/16 instance as synthetic candidate scoring.
            self._clip_model = get_clip_model(CLIP_MODEL_NAME, device)

        # 1) semantic prior S: CLIP text-embedding similarity
        templates = self._clip_templates()
        old_text = build_clip_text_features(
            self._clip_model, CLIP_MODEL_NAME,
            [self._class_name_for(c) for c in old_classes], device, templates,
        )
        new_text = build_clip_text_features(
            self._clip_model, CLIP_MODEL_NAME,
            [self._class_name_for(c) for c in new_classes], device, templates,
        )
        S_matrix = old_text @ new_text.T

        # 2) visual confusion Q: how strongly new-class synthetic images fire on old logits
        probe_per_class = int(self.args.get("calib_vis_probes_per_class", 50))
        Q_matrix = torch.zeros((len(old_classes), len(new_classes)), device=device)
        eval_tf = self._net_eval_transform_pil()
        self._network.eval()

        for j, n_cls in enumerate(new_classes):
            paths = self._get_current_task_syn_paths_for_class(n_cls)[:probe_per_class]
            if not paths:
                continue
            imgs = []
            for p in paths:
                with Image.open(p) as im:
                    imgs.append(eval_tf(im.convert("RGB")))
            out = self._network(torch.stack(imgs).to(device), self.args.get("perturb_var", None))
            probs_old = F.softmax(out["logits"][:, : self._known_classes], dim=1)
            Q_matrix[:, j] = probs_old.mean(dim=0)

        # 3) non-linear two-factor gating
        alpha, beta = 15.0, 20.0                       # sigmoid sharpness
        base_noise = 1.0 / max(1, len(old_classes))    # chance level on the old head
        dynamic_vis_tau = base_noise + self.calib_vis_tau

        G_sem = torch.sigmoid(alpha * (S_matrix - self.calib_sem_tau))
        G_vis = torch.sigmoid(beta * (Q_matrix - dynamic_vis_tau))
        self.gate_matrix = G_sem * G_vis

        max_idx = int(torch.argmax(self.gate_matrix))
        o_idx, n_idx = divmod(max_idx, len(new_classes))
        logging.info(
            "[CVS-GC] task %d | avg gate %.4f | max gate %.4f | strongly gated pairs %d",
            self._cur_task,
            self.gate_matrix.mean().item(),
            self.gate_matrix.max().item(),
            int((self.gate_matrix > 0.5).sum().item()),
        )
        logging.info(
            "[CVS-GC] riskiest pair: old '%s' <-> new '%s'",
            self._class_name_for(old_classes[o_idx]),
            self._class_name_for(new_classes[n_idx]),
        )

    def _get_synthetic_old_images(self, old_classes, batch_size=8):
        """Sample one cached probe image per requested old class."""
        if not self._cached_syn_gpu:
            return None, None

        imgs, labs = [], []
        for old_c in old_classes:
            bank = self._cached_syn_gpu.get(old_c, None)
            if bank is None or bank.size(0) == 0:
                continue
            j = random.randrange(bank.size(0))
            imgs.append(bank[j])
            labs.append(old_c)

        if not imgs:
            return None, None

        if len(imgs) > batch_size:
            pick = random.sample(range(len(imgs)), k=batch_size)
            imgs = [imgs[i] for i in pick]
            labs = [labs[i] for i in pick]

        return torch.stack(imgs, dim=0), torch.tensor(labs, dtype=torch.long, device=self._device)

    # ------------------------------------------------------------------ #
    # training entry point
    # ------------------------------------------------------------------ #
    def incremental_train(self, data_manager):
        self._cur_task += 1
        task_token = self._timing_begin()

        if self.args.get("need_syn_imgs", False):
            with self._timed("generator_init"):
                self.generator_init(self._cur_task)
        else:
            self._generator = None

        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on %d-%d", self._known_classes, self._total_classes)

        setup_token = self._timing_begin()
        self.data_manager = data_manager
        self.train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train", mode="train", kshot=self.args["kshot"],
        )

        is_fs = isinstance(self.args["kshot"], int) and self._known_classes > 0
        train_bs = self.args["fs_batch_size"] if is_fs else self.batch_size
        eval_bs = int(self.args.get("eval_batch_size", self.batch_size))
        train_workers = self._worker_count("train_num_workers", 0 if self._known_classes > 0 else DEFAULT_NUM_WORKERS)
        test_workers = self._worker_count("test_num_workers", DEFAULT_NUM_WORKERS)
        proto_workers = self._worker_count("protonet_num_workers", 0)

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=train_bs, shuffle=True,
            num_workers=train_workers, pin_memory=True,
        )
        self.local_cur_loaders.append(self.train_loader)

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(
            test_dataset, batch_size=eval_bs, shuffle=False, num_workers=test_workers, pin_memory=True
        )

        test_curr_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="test", mode="test"
        )
        self.test_curr_loader = DataLoader(
            test_curr_dataset, batch_size=eval_bs, shuffle=False, num_workers=test_workers, pin_memory=True
        )

        self.train_dataset_for_protonet = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train", mode="test", kshot=self.args["kshot"],
        )
        self.train_loader_for_protonet = DataLoader(
            self.train_dataset_for_protonet, batch_size=self.batch_size,
            shuffle=False, num_workers=proto_workers,
        )

        logging.info(
            "training set size: %d, fc construct set size: %d",
            len(self.train_dataset), len(self.train_dataset_for_protonet),
        )
        self._timing_end(
            setup_token, "dataset_loader_setup",
            train_size=int(len(self.train_dataset)), test_size=int(len(test_dataset)),
            train_bs=int(train_bs), eval_bs=int(eval_bs),
        )

        if len(self._multiple_gpus) > 1:
            logging.info("Using %d GPUs via DataParallel", len(self._multiple_gpus))
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self.all_classes = np.unique(self.train_dataset.labels)
        self.min_class_id = int(np.min(self.all_classes))
        self.max_class_id = int(np.max(self.all_classes))

        if self.args.get("need_syn_imgs", False):
            self._generate_synthetic_images()
        self._validate_synthetic_cache()
        with self._timed("init_syn_dataloader"):
            self._init_syn_dataloader()

        with self._timed("train_session_total"):
            self._train(self.train_loader, self.test_loader,
                        self.train_loader_for_protonet, self.local_cur_loaders)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        self._timing_end(
            task_token, "task_total",
            need_syn_imgs=bool(self.args.get("need_syn_imgs", False)),
            train_size=int(len(self.train_dataset)),
            total_seen_classes=int(self._total_classes),
        )

    def _generate_synthetic_images(self):
        """Class inversion followed by latent-diffusion sampling for this task."""
        gen_setup_token = self._timing_begin()
        gen_dataset = GenDataset(
            images=self.train_dataset.images,
            class_ids=self.train_dataset.labels,
            use_path=getattr(self.data_manager, "use_path", False),
            min_class_id=self._known_classes,
            class_names=self._get_class_names(),
            dataset_name=self.args.get("dataset", "cifar224"),
            syn_prompt_mode=self.args.get("syn_prompt_mode", "default"),
            syn_prompt_fallback_template=self.args.get(
                "syn_prompt_fallback_template", self.args.get("synthesis_prompt", "a photo of *")
            ),
        )
        gen_loader = DataLoader(
            deepcopy(gen_dataset),
            batch_size=self.args["g_local_bs"],
            num_workers=self._worker_count("gen_num_workers", 0),
            shuffle=True,
        )
        self.gen_data_iter = DataIter(gen_loader)
        self._timing_end(
            gen_setup_token, "generator_data_setup",
            gen_dataset_size=int(len(gen_dataset)),
            gen_batch_size=int(self.args["g_local_bs"]),
        )

        with self._timed(
            "class_inversion_total",
            com_round_gen=int(self.args.get("com_round_gen", 0)),
            g_local_train_steps=int(self.args.get("g_local_train_steps", 0)),
        ):
            inv_text_embeds = self._class_inversion()

        syn_token = self._timing_begin()
        n_generated = self._synthesis_imgs(inv_text_embeds)
        self._timing_end(
            syn_token, "synthesis_imgs_total",
            generated_images=int(n_generated), n_iter=int(self.args.get("n_iter", 0)),
        )

        # Newly written files invalidate the cached path index.
        self._syn_path_index = None

    def _validate_synthetic_cache(self):
        """Fail early when a replay-enabled run has no candidates for this task."""
        if self._cur_task == 0:
            replay_required = bool(self.args.get("use_base_syn", True))
        else:
            replay_required = (
                int(self.args.get("protonet_syn_per_class", 0)) > 0
                or bool(self.args.get("use_old_probes", True))
                or float(self.args.get("calib_lambda", 0.0)) > 0.0
            )
        if not replay_required:
            return

        missing = [
            int(class_id) for class_id in self.all_classes
            if not self._get_current_task_syn_paths_for_class(int(class_id))
        ]
        if missing:
            raise FileNotFoundError(
                "Synthetic replay is enabled, but task_{} has no cached images for class(es) {} "
                "under {!r}. Set need_syn_imgs=true to generate them, or point "
                "syn_image_path to a complete cache.".format(
                    self._cur_task, missing[:10], self.syn_imgs_dir
                )
            )

    def _train(self, train_loader, test_loader, train_loader_for_protonet, local_cur_loaders):
        self._network.to(self._device)
        self.rule_scorer.to(self._device)

        total_params = sum(p.numel() for p in self._network.parameters())
        trainable_params = sum(p.numel() for p in self._network.parameters() if p.requires_grad)
        logging.info("total parameters: %d", total_params)
        logging.info("trainable parameters: %d", trainable_params)

        if self._cur_task > 0:
            self._build_old_probe_cache()

            with self._timed("protonet_aug_build", stage="pre_rebuild"):
                protonet_aug_loader = self._build_protonet_loader_with_selected_syn(train_loader_for_protonet)
            with self._timed("ema_prompt_update", stage="pre_rebuild"):
                self.update_ema_prompt(protonet_aug_loader)
            with self._timed("replace_fc", stage="pre_rebuild"):
                self.replace_fc(protonet_aug_loader, self._network)

        # ---- optimizer / scheduler (incremental sessions use fs_lr / fs_epoch) ----
        is_fs = isinstance(self.args.get("kshot", None), int) and self._known_classes > 0
        lr_main = float(self.args.get("fs_lr", self.init_lr)) if is_fs else float(self.init_lr)
        tmax = max(1, int(self.args["fs_epoch"]) if is_fs else int(self.args["tuned_epoch"]))

        # Freezing the head during few-shot finetuning keeps the prototypes stable.
        if self._cur_task > 0 and bool(self.args.get("fs_freeze_fc", True)):
            self._network.fc.weight.requires_grad_(False)

        param_groups = [{"params": [p for p in self._network.parameters() if p.requires_grad]}]
        if self._cur_task == 0:
            # Both the curation mask and the rule scorer are learned in the base session only.
            if self.mask_bank is not None:
                param_groups.append({
                    "params": self.mask_bank.parameters(),
                    "lr": lr_main * float(self.args.get("mask_lr_mul", 0.1)),
                    "weight_decay": 0.0,
                })
            param_groups.append({
                "params": self.rule_scorer.parameters(),
                "lr": lr_main,
                "weight_decay": 0.0,
            })

        if self.args["optimizer"] == "sgd":
            optimizer = optim.SGD(param_groups, momentum=0.9, lr=lr_main, weight_decay=self.weight_decay)
        elif self.args["optimizer"] == "adam":
            optimizer = optim.AdamW(param_groups, lr=lr_main, weight_decay=self.weight_decay)
        else:
            raise ValueError("Unsupported optimizer: {}".format(self.args["optimizer"]))

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmax, eta_min=self.min_lr)

        if self._cur_task > 0:
            with self._timed("calibration_gate_prepare"):
                self._prepare_calibration_gates()

        if self._cur_task == 0:
            # ``use_base_syn=False`` physically cuts synthetic images out of the base mix.
            use_mixed = bool(self.args.get("use_base_syn", True)) and bool(local_cur_loaders)
            grad_loader = local_cur_loaders[0] if use_mixed else train_loader
        else:
            grad_loader = train_loader

        with self._timed(
            "train_loop_total", total_epoch=int(tmax),
            train_batches=int(len(grad_loader)),
        ):
            self._init_train(grad_loader, test_loader, optimizer, scheduler)

        if self._cur_task == 0:
            with self._timed("save_base_model"):
                torch.save(self._network.state_dict(), self.args["base_model_path"])
            with self._timed("ema_prompt_update", stage="base_finalize"):
                self.update_ema_prompt(self.train_loader_for_protonet, mode="base")
            with self._timed("replace_fc", stage="base_finalize"):
                self.replace_fc(self.train_loader_for_protonet, self._network)
        else:
            with self._timed("protonet_aug_build", stage="final_rebuild"):
                protonet_aug_loader = self._build_protonet_loader_with_selected_syn(train_loader_for_protonet)
            with self._timed("ema_prompt_update", stage="final_rebuild"):
                self.update_ema_prompt(protonet_aug_loader)
            with self._timed("replace_fc", stage="final_rebuild"):
                self.replace_fc(protonet_aug_loader, self._network)

    def _build_old_probe_cache(self):
        """Cache a few synthetic images per old class on the GPU.

        They act as cheap stand-ins for real old data in the reverse branch of
        the calibration loss; no real old samples are stored.
        """
        cache_token = self._timing_begin()
        self._cached_syn_gpu = {}
        if not self.use_old_probes:
            self._timing_end(cache_token, "old_probe_cache_build", use_old_probes=False)
            return

        k_per_class = int(self.args.get("old_probe_k_per_class", 10))
        tf = self._net_eval_transform_pil()

        for c in range(self._known_classes):
            paths = self._get_syn_paths_for_class(c)
            if not paths:
                self._cached_syn_gpu[c] = None
                continue

            tensors = []
            for p in random.sample(paths, k=min(len(paths), k_per_class)):
                try:
                    with Image.open(p) as im:
                        tensors.append(tf(im.convert("RGB")))
                except Exception as exc:
                    logging.warning("Failed to load old-class probe %s: %s", p, exc)

            self._cached_syn_gpu[c] = (
                torch.stack(tensors, dim=0).to(self._device, non_blocking=True) if tensors else None
            )

        cached_classes = sum(1 for v in self._cached_syn_gpu.values() if v is not None)
        cached_images = sum(int(v.size(0)) for v in self._cached_syn_gpu.values() if v is not None)
        logging.info("[OldProbe] cached %d images across %d old classes", cached_images, cached_classes)
        self._timing_end(
            cache_token, "old_probe_cache_build",
            use_old_probes=True, cached_probe_classes=int(cached_classes),
            cached_probe_images=int(cached_images), k_per_class=int(k_per_class),
        )

    # ------------------------------------------------------------------ #
    # inner training loop
    # ------------------------------------------------------------------ #
    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        is_fs = isinstance(self.args.get("kshot", None), int) and self._known_classes > 0
        total_epoch = max(0, int(self.args["fs_epoch"]) if is_fs else int(self.args["tuned_epoch"]))
        ep_name = "FS-Epoch" if is_fs else "Epoch"

        # base-session mask hyper-parameters
        beta = float(self.args.get("mask_reg_beta", 2.0))
        lambda_div = float(self.args.get("clip_diversity_lambda", 0.1))
        sel_ratio = float(self.args.get("clip_selection_ratio", 0.15))
        reg_w = float(self.args.get("mask_reg_weight", 1.0))

        anchor_refresh = int(self.args.get("base_anchor_refresh_interval", 1))
        proto_refresh = int(self.args.get("base_proto_refresh_interval", 1))
        anchor_samples = None
        base_real_proto_bank = None

        for epoch in range(total_epoch):
            epoch_token = self._timing_begin()
            data_wait_wall = step_wall = eval_wall = 0.0

            if self._cur_task == 0:
                self._reset_mask_log()

                if anchor_samples is None or (anchor_refresh > 0 and epoch % anchor_refresh == 0):
                    anchor_samples = self.find_anchor_sample(self._network, self.train_loader_for_protonet)
                if self.rule_mode == "3d" and (
                    base_real_proto_bank is None or (proto_refresh > 0 and epoch % proto_refresh == 0)
                ):
                    base_real_proto_bank = self._compute_real_class_prototypes(self.train_loader_for_protonet)

            self._network.train()
            losses = 0.0
            correct = total = 0
            epoch_fwd_pushes = epoch_rev_pushes = 0
            epoch_calib_loss = epoch_gap = 0.0

            iter_start = time.perf_counter()
            for gidx, inputs, targets in train_loader:
                data_wait_wall += time.perf_counter() - iter_start
                step_start = time.perf_counter()
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                if self._cur_task == 0:
                    loss, logits_for_acc, mask_batch = self._base_step(
                        gidx.to(self._device), inputs, targets, anchor_samples,
                        base_real_proto_bank, epoch, beta, lambda_div, sel_ratio, reg_w,
                    )
                else:
                    loss, logits_for_acc, stats = self._incremental_step(inputs, targets)
                    epoch_fwd_pushes += stats["fwd_pushes"]
                    epoch_rev_pushes += stats["rev_pushes"]
                    epoch_calib_loss += stats["calib_loss"]
                    epoch_gap += stats["logit_gap"]
                    mask_batch = None

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += float(loss.item())
                with torch.no_grad():
                    correct += int(logits_for_acc.argmax(dim=1).eq(targets).sum().item())
                    total += int(targets.size(0))

                if self._cur_task == 0 and mask_batch is not None:
                    self._accumulate_mask_log(gidx.to(self._device), mask_batch)

                step_wall += time.perf_counter() - step_start
                iter_start = time.perf_counter()

            scheduler.step()

            train_acc = float(correct) * 100.0 / max(1, total)
            if self._eval_each_epoch or (epoch + 1 == total_epoch):
                eval_start = time.perf_counter()
                test_cur_acc = self._compute_accuracy(self._network, self.test_curr_loader)
                test_acc = self._compute_accuracy(self._network, test_loader)
                self._sync_for_timing()
                eval_wall = time.perf_counter() - eval_start
                eval_msg = "Cur: {:.2f}, All: {:.2f}".format(test_cur_acc, test_acc)
            else:
                eval_msg = "Cur: skipped, All: skipped"

            info = "Task {}, {} {} => Loss {:.3f}, Train: {:.2f}, {}".format(
                self._cur_task, ep_name, epoch + 1,
                losses / max(1, len(train_loader)), train_acc, eval_msg,
            )
            if self._cur_task > 0:
                info += " | Fwd Push: {}, Rev Push: {}".format(epoch_fwd_pushes, epoch_rev_pushes)
            logging.info(info)

            if self._cur_task > 0 and self.gate_matrix is not None:
                n_batches = max(1, len(train_loader))
                logging.info(
                    "   [CVS-GC] L_calib=%.4f | avg logit gap (new - hardest old)=%.3f",
                    epoch_calib_loss / n_batches, epoch_gap / n_batches,
                )
            if self._cur_task == 0:
                self._log_mask_epoch(epoch)

            self._timing_end(
                epoch_token, "train_epoch",
                epoch=int(epoch + 1), total_epoch=int(total_epoch), is_fs=bool(is_fs),
                batches=int(len(train_loader)), samples=int(total),
                loss=float(losses / max(1, len(train_loader))), train_acc=float(train_acc),
                data_wait_wall_sec=float(data_wait_wall), step_wall_sec=float(step_wall),
                eval_wall_sec=float(eval_wall),
                fwd_pushes=int(epoch_fwd_pushes), rev_pushes=int(epoch_rev_pushes),
            )

    def _base_step(self, gidx, inputs, targets, anchor_samples, base_real_proto_bank,
                   epoch, beta, lambda_div, sel_ratio, reg_w):
        """One base-session step: masked CE + anchor alignment + KL + mask losses."""
        real_flags = (gidx < self._real_len).float()
        syn_flags = 1.0 - real_flags
        syn_count = syn_flags.sum().clamp(min=1.0)

        if self.mask_bank is not None and reg_w > 0.0:
            x = self.mask_bank.soft(gidx)                # continuous keep probability
            x_st = self.mask_bank.hard_ste(x)            # binarised, straight-through
            mask_batch = real_flags + syn_flags * x      # real samples are never masked out
        else:
            x = x_st = None
            mask_batch = torch.ones_like(real_flags)

        # Append one anchor image per class present in the batch.
        cur_class = sorted(set(targets.cpu().tolist()))
        n_anchor = len(cur_class)
        anchor_inputs = torch.cat(
            [inputs] + [anchor_samples[c].unsqueeze(0).to(self._device) for c in cur_class], dim=0
        )

        out = self._network(anchor_inputs, self.args["perturb_var"])
        logits = out["logits"][:-n_anchor, :]
        features = out["features"]
        mu, std = out["kl"]

        ce_vec = F.cross_entropy(logits, targets, reduction="none")
        loss_cls = (ce_vec * mask_batch).sum() / mask_batch.sum().clamp(min=1.0)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        sim_loss = 0.0
        for anchor_id, c in enumerate(cur_class):
            fea_c = features[:-n_anchor][targets == c]
            # NOTE: indexing preserved verbatim from the original implementation.
            # The anchors are the *last* ``n_anchor`` rows, so a strict reading
            # would be ``features[-n_anchor:][anchor_id]``. Changing it alters
            # published numbers, so the original slice is kept.
            fea_anchor = features[n_anchor:][anchor_id].detach().unsqueeze(0).repeat(len(fea_c), 1)
            sim_loss = sim_loss + (1 - cos(fea_c, fea_anchor)).mean()
        sim_loss = sim_loss / max(1, n_anchor)

        kl = 0.5 * torch.sum(mu.pow(2) + std.pow(2) - 2 * std.log() - 1) / mu.size(0)
        loss = (
            loss_cls
            + float(self.args["anchor_lambda"]) * sim_loss
            + float(self.args["kl_weight"]) * kl
        )

        if x is not None:
            # Mask objective: keep semantically faithful and diverse samples,
            # while a budget term pins the kept fraction at ``sel_ratio``.
            sim_b = self.sim_scores[gidx].float()
            div_b = self.div_scores[gidx].float()
            sim_mean = (sim_b * syn_flags).sum() / syn_count
            div_mean = (div_b * syn_flags).sum() / syn_count

            loss_sim = -((x * (sim_b / (sim_mean + 1e-8))) * syn_flags).sum() / syn_count
            loss_div = -((x * (div_b / (div_mean + 1e-8))) * syn_flags).sum() / syn_count * lambda_div
            loss_budget = torch.abs(
                ((x_st * syn_flags).sum() - sel_ratio * syn_count) / syn_count
            ) * beta
            loss = loss + reg_w * (loss_sim + loss_div + loss_budget)

            # Distil the converged mask into the transferable rule scorer.
            if self.rule_distill_weight > 0.0 and epoch >= self.rule_warmup_epochs:
                z_rule, syn_mask = self._build_rule_input_from_batch(
                    gidx=gidx, targets=targets,
                    feat_actual=features[:-n_anchor] if n_anchor > 0 else features,
                    syn_flags=syn_flags, real_proto_dict=base_real_proto_bank,
                )
                if z_rule is not None:
                    # Detach both sides: the rule learns the mapping, not the mask.
                    x_rule = torch.sigmoid(self.rule_tau * self.rule_scorer(z_rule.detach()).squeeze(1))
                    loss = loss + self.rule_distill_weight * F.mse_loss(x_rule, x[syn_mask].detach())

        return loss, logits, mask_batch

    def _incremental_step(self, inputs, targets):
        """One incremental step: masked CE + bidirectional gated calibration."""
        out = self._network(inputs, self.args["perturb_var"])
        logits = out["logits"][:, : self._total_classes]

        # 1) few-shot CE restricted to the new classes
        ce_logits = logits.clone()
        ce_logits[:, : self._known_classes] = float("-inf")
        loss_ce = F.cross_entropy(ce_logits, targets)

        loss_calib = torch.tensor(0.0, device=self._device)
        stats = {"fwd_pushes": 0, "rev_pushes": 0, "calib_loss": 0.0, "logit_gap": 0.0}

        if self.gate_matrix is not None:
            # 2a) forward push (new -> old): stop new classes invading old territory
            new_targets_idx = targets - self._known_classes
            max_old_logits, max_old_idx = torch.max(logits[:, : self._known_classes], dim=1)

            gates_fwd = self.gate_matrix[max_old_idx, new_targets_idx]
            margins_fwd = self.calib_m0 + self.calib_m1 * gates_fwd
            z_new = logits[torch.arange(logits.size(0)), targets]

            penalty_fwd = F.relu(max_old_logits - z_new + margins_fwd)
            loss_calib_fwd = (gates_fwd * penalty_fwd).mean()
            stats["fwd_pushes"] = int((penalty_fwd > 0).sum().item())
            stats["logit_gap"] = float((z_new - max_old_logits).mean().item())

            # 2b) reverse push (old -> new): keep cached old probes from being absorbed
            loss_calib_rev = torch.tensor(0.0, device=self._device)
            if self._known_classes > 0 and self.use_old_probes:
                sampled_old = random.sample(
                    range(self._known_classes), min(self._known_classes, inputs.size(0))
                )
                syn_old_inputs, syn_old_targets = self._get_synthetic_old_images(
                    sampled_old, batch_size=inputs.size(0)
                )
                if syn_old_inputs is not None:
                    out_old = self._network(syn_old_inputs, self.args.get("perturb_var", None))
                    logits_old = out_old["logits"][:, : self._total_classes]

                    z_old = logits_old[torch.arange(logits_old.size(0)), syn_old_targets]
                    max_new_logits, _ = torch.max(
                        logits_old[:, self._known_classes : self._total_classes], dim=1
                    )
                    # Ungated hinge: every old class is at risk, so use the base margin.
                    penalty_rev = F.relu(max_new_logits - z_old + self.calib_m0)
                    loss_calib_rev = penalty_rev.mean()
                    stats["rev_pushes"] = int((penalty_rev > 0).sum().item())

            loss_calib = (
                float(self.args.get("calib_fwd_weight", 0.5)) * loss_calib_fwd
                + float(self.args.get("calib_rev_weight", 0.01)) * loss_calib_rev
            )
            stats["calib_loss"] = float(loss_calib.item())

        return loss_ce + self.calib_lambda * loss_calib, logits, stats

    # ------------------------------------------------------------------ #
    # prompt EMA and anchors
    # ------------------------------------------------------------------ #
    def update_ema_prompt(self, train_loader, mode="new"):
        """Update the task-invariant prompt with an EMA over instance prompts."""
        self._network.eval()
        ignore_syn = bool(self.args.get("ema_prompt_ignore_syn", True))
        ds = train_loader.dataset
        has_syn_flag = hasattr(ds, "is_syn")

        prompt_list = []
        with torch.no_grad():
            for idx, data, label in train_loader:
                if ignore_syn and has_syn_flag:
                    keep = [ds.is_syn[i] == 0 for i in idx.cpu().tolist()]
                    if not any(keep):
                        continue
                    data = data[torch.tensor(keep, device=data.device)]

                data = data.to(self._device)
                prompt, _ = self._network.backbone.Prompt_Encoder(data, self._network.backbone.TIP, 0)
                prompt_list.append(prompt.detach().cpu())

        if not prompt_list:
            return

        mean_prompt = torch.mean(torch.cat(prompt_list, dim=0), dim=0)
        if mode == "new":
            beta = self.args["EMA_beta"]
            self._network.backbone.Avg_TSP = beta * self._network.backbone.Avg_TSP + (1 - beta) * mean_prompt
        else:
            self._network.backbone.Avg_TSP = mean_prompt
        self._network.backbone.Avg_TSP.to(self._device)

    def find_anchor_sample(self, model, train_loader):
        """Pick, per class, the real sample closest to that class' feature mean."""
        model.eval()
        embedding_list, label_list, prompt_list = [], [], []

        with torch.no_grad():
            for _, data, label in train_loader:
                data = data.to(self._device)
                embedding_list.append(model(data)["features"].cpu())
                label_list.append(label.cpu())
                prompt, _ = self._network.backbone.Prompt_Encoder(data, self._network.backbone.TIP, 0)
                prompt_list.append(prompt.detach().cpu())

        embeddings = torch.cat(embedding_list, dim=0)
        labels = torch.cat(label_list, dim=0)
        self._network.backbone.Avg_TSP = torch.mean(torch.cat(prompt_list, dim=0), dim=0)
        self._network.backbone.Avg_TSP.to(self._device)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        anchor_by_class = {}
        for class_index in np.unique(labels.numpy()):
            data_index = (labels == int(class_index)).nonzero().squeeze(-1)
            emb = embeddings[data_index]
            class_mean = emb.mean(0).unsqueeze(0).repeat(len(emb), 1)
            anchor_index = torch.argmax(cos(emb, class_mean))
            anchor_by_class[int(class_index)] = train_loader.dataset[data_index[anchor_index]][1]
        return anchor_by_class

    # ------------------------------------------------------------------ #
    # latent diffusion: class inversion and sampling
    # ------------------------------------------------------------------ #
    def generator_init(self, cur_task):
        # Imported lazily so that runs with ``need_syn_imgs: false`` (i.e. reading
        # pre-generated images from disk) do not need the diffusion stack installed.
        from ldm import LatentDiffusion
        from omegaconf import OmegaConf

        if not os.path.isfile(self.args["config"]):
            raise FileNotFoundError("LDM config not found: {}".format(self.args["config"]))
        if not os.path.isfile(self.args["ldm_ckpt"]):
            raise FileNotFoundError(
                "LDM checkpoint not found: {}. Download it or set ldm_ckpt in the config.".format(
                    self.args["ldm_ckpt"]
                )
            )
        bert_vocab = os.path.join("models", "bert", "vocab.txt")
        if not os.path.isfile(bert_vocab):
            raise FileNotFoundError(
                "Local BERT tokenizer not found at {}. Download "
                "google-bert/bert-base-uncased into models/bert as documented in README.md.".format(
                    bert_vocab
                )
            )

        self.config = OmegaConf.load(self.args["config"])
        self.config.model.params.ckpt_path = self.args["ldm_ckpt"]
        num_classes = self.args["init_cls"] if cur_task == 0 else self.args["increment"]
        self.config["model"]["params"]["personalization_config"]["params"]["num_classes"] = num_classes

        self._generator = LatentDiffusion(**self.config["model"]["params"])
        self._generator.load_state_dict(
            torch.load(self.args["ldm_ckpt"], map_location="cpu")["state_dict"], strict=False
        )
        self.generator_init_embedding = deepcopy(self._generator.embedding_manager.state_dict())
        self._generator.learning_rate = (
            self.config.data.params.batch_size * self.config.model.base_learning_rate
        )
        logging.info(
            "Generator LR = %.2e = %d (batch size) * %.2e (base lr)",
            self._generator.learning_rate,
            self.config.data.params.batch_size,
            self.config.model.base_learning_rate,
        )

    def _local_update_g(self, generator, gen_data_loader):
        generator.train()
        generator = generator.to(self._device)
        optimizer = generator.configure_optimizers()
        for _ in range(self.args["g_local_train_steps"]):
            batch = gen_data_loader.next()
            batch["image"] = batch["image"].to(self._device)
            loss, _ = generator.shared_step(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return generator.embedding_manager.state_dict()

    def _class_inversion(self):
        """Learn one text embedding per new class from its k-shot real images."""
        self._generator.to(self._device)
        # Always restart from the pretrained embedding so tasks stay independent.
        self._generator.embedding_manager.load_state_dict(deepcopy(self.generator_init_embedding))

        if self.gen_data_iter is None:
            raise RuntimeError("No generator data iterator available for class inversion.")

        # Each round restarts the optimizer, which acts as a mild reset schedule
        # over the ``g_local_train_steps`` inner steps.
        for round_idx in tqdm(range(self.args["com_round_gen"]), desc="Class Inversion"):
            round_token = self._timing_begin()
            local_generator = deepcopy(self._generator)
            weights = self._local_update_g(local_generator, self.gen_data_iter)
            self._generator.embedding_manager.load_state_dict(deepcopy(weights))
            self._timing_end(round_token, "class_inversion_round", round=int(round_idx + 1))

        if self.args.get("save_cls_embeds", False):
            ckpt_path = os.path.join(
                self.args["save_dir"], "cls_embeds_ckpt",
                "{}-{}_embedding_manager.pt".format(self.min_class_id, self.max_class_id),
            )
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(self._generator.embedding_manager.state_dict(), ckpt_path)

        return deepcopy(self._generator.embedding_manager.string_to_param_dict)

    def _synthesis_imgs(self, inv_text_embeds):
        """Sample images for every class of the current task and write them to disk.

        Files are laid out as ``<syn_image_path>/task_<t>/<class_id>/<class_id>-<i>.jpg``,
        which is the layout every reader in this file assumes.
        """
        from einops import rearrange
        from ldm import DDIMSampler

        if inv_text_embeds is not None:
            self._generator.embedding_manager.string_to_param_dict = inv_text_embeds

        sampler = DDIMSampler(self._generator)
        outdir = os.path.join(self.syn_imgs_dir, "task_{}".format(self._cur_task))
        os.makedirs(outdir, exist_ok=True)

        n_samples = int(self.args.get("synthesis_n_samples", 40))
        n_iter = int(self.args["n_iter"])
        ddim_steps = int(self.args.get("ddim_steps", 50))
        scale = float(self.args.get("synthesis_guidance_scale", 10.0))
        ddim_eta = float(self.args.get("ddim_eta", 0.0))
        latent_shape = [4, 256 // 8, 256 // 8]

        n_generated = 0
        with torch.no_grad(), self._generator.ema_scope():
            for tmp_cls in self.all_classes:
                class_token = self._timing_begin()
                class_dir = os.path.join(outdir, str(int(tmp_cls)))
                os.makedirs(class_dir, exist_ok=True)
                prompts = self._synthesis_prompts_for_class(int(tmp_cls))

                rel_cls = torch.LongTensor([int(tmp_cls) - self.min_class_id] * n_samples)
                uc = None
                if scale != 1.0:
                    uc = self._generator.get_learned_conditioning(n_samples * [""], rel_cls)

                base_count = 0
                for _ in trange(n_iter, desc="Sampling class {}".format(int(tmp_cls))):
                    prompt = random.choice(prompts)
                    c = self._generator.get_learned_conditioning(n_samples * [prompt], rel_cls)
                    samples_ddim, _ = sampler.sample(
                        S=ddim_steps, conditioning=c, batch_size=n_samples,
                        shape=latent_shape, verbose=False,
                        unconditional_guidance_scale=scale, unconditional_conditioning=uc,
                        eta=ddim_eta,
                    )
                    x_samples = self._generator.decode_first_stage(samples_ddim)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                    for x_sample in x_samples:
                        arr = 255.0 * rearrange(x_sample.cpu().numpy(), "c h w -> h w c")
                        Image.fromarray(arr.astype(np.uint8)).save(
                            os.path.join(class_dir, "{}-{}.jpg".format(int(tmp_cls), base_count))
                        )
                        base_count += 1

                n_generated += base_count
                self._timing_end(
                    class_token, "synthesis_imgs_class",
                    class_id=int(tmp_cls), generated_images=int(base_count),
                    n_iter=n_iter, n_samples=n_samples, ddim_steps=ddim_steps,
                    guidance_scale=scale, ddim_eta=ddim_eta,
                    prompt_variants=int(len(set(prompts))),
                )
        return n_generated

    def _synthesis_prompts_for_class(self, class_id):
        """Prompts for one class, optionally matched to ImageNet-R filename styles."""
        fallback = str(self.args.get("synthesis_prompt", "a photo of *"))
        dataset_name = str(self.args.get("dataset", "")).lower()
        class_name = self._class_name_for(class_id)

        if dataset_name == "imagenetr" and str(
            self.args.get("syn_prompt_mode", "default")
        ).lower() == "prefix":
            paths = [
                str(path)
                for path, label in zip(self.train_dataset.images, self.train_dataset.labels)
                if int(label) == int(class_id)
            ]
            return imagenetr_prompts_for_paths(
                paths, class_name,
                fallback=self.args.get("syn_prompt_fallback_template", "a rendition of *"),
            )

        mode = str(self.args.get("synthesis_prompt_mode", "placeholder")).lower()
        if mode == "auto":
            mode = "placeholder" if self._cur_task == 0 else "hybrid"
        if mode == "placeholder":
            return [fallback]
        if mode == "class_name":
            template = self.args.get(
                "synthesis_prompt_template", "a photo of {article} {class_name}"
            )
        elif mode == "hybrid":
            template = self.args.get(
                "synthesis_hybrid_prompt_template",
                "a photo of {article} {class_name} {placeholder}",
            )
        else:
            raise ValueError(
                "synthesis_prompt_mode must be placeholder, class_name, hybrid, or auto; "
                "got {!r}.".format(mode)
            )
        return [format_class_synthesis_prompt(template, class_name)]

    # ------------------------------------------------------------------ #
    # synthetic data loaders
    # ------------------------------------------------------------------ #
    def _init_syn_dataloader(self):
        """Base session only: build the real+synthetic mix and score it with CLIP.

        Incremental sessions consume synthetic data exclusively through
        ``_build_protonet_loader_with_selected_syn``, so all mask state is reset.
        """
        if self._cur_task > 0:
            self.local_cur_loaders = []
            self.sim_scores = None
            self.div_scores = None
            self.mask_bank = None
            return

        syn_token = self._timing_begin()
        class_ids = sorted(set(int(x) for x in self.all_classes))
        paths_by_class = {
            c: self._get_current_task_syn_paths_for_class(c) for c in class_ids
        }
        candidate_paths = [path for c in class_ids for path in paths_by_class[c]]

        # Score the complete generated pool before applying ``cur_size``.  The
        # old order did the opposite and therefore only scored filenames
        # 0..cur_size-1, leaving most generated candidates unused.
        preselected_sim = preselected_div = None
        if bool(self.args.get("base_clip_preselect", True)):
            to_tensor = transforms.ToTensor()
            class_names = self._get_class_names()
            keep_per_class = int(self.args["cur_size"])
            chosen_paths, sim_chunks, div_chunks = [], [], []

            for class_id in class_ids:
                class_paths = paths_by_class[class_id]
                class_tensors = []
                for path in class_paths:
                    with Image.open(path) as image:
                        class_tensors.append(to_tensor(image.convert("RGB")))
                if not class_tensors:
                    continue

                class_labels = [class_id] * len(class_tensors)
                sim, div = compute_clip_scores_memory(
                    class_tensors, class_labels, [class_names[class_id]],
                    device=self._device, templates=self._clip_templates(),
                )
                combined = sim + float(self.args.get("clip_diversity_lambda", 0.1)) * div
                keep = min(keep_per_class, len(class_paths))
                top_indices = torch.topk(combined, k=keep, largest=True).indices
                top_list = top_indices.tolist()
                chosen_paths.extend(class_paths[index] for index in top_list)
                sim_chunks.append(sim[top_indices])
                div_chunks.append(div[top_indices])

            candidate_paths = chosen_paths
            if sim_chunks:
                preselected_sim = torch.cat(sim_chunks).float()
                preselected_div = torch.cat(div_chunks).float()
            logging.info(
                "[BasePreselect] retained %d/%d candidates (%d per class cap)",
                len(candidate_paths), sum(len(paths) for paths in paths_by_class.values()),
                keep_per_class,
            )

        cur_syn_dataset = TaskSynImageDataset(
            candidate_paths,
            size_per_cls=self.args["cur_size"],
            selection_mode=self.args.get("cur_size_sampling", "head"),
            transform=syn_train_transform(self.args["dataset_syn"]),
        )
        self._timing_end(
            syn_token, "syn_dataset_load",
            syn_dataset_size=int(len(cur_syn_dataset)), cur_size=int(self.args["cur_size"]),
        )

        mixed_token = self._timing_begin()
        mixed_ds, total_len = build_indexed_concat([self.train_dataset, cur_syn_dataset])
        self._real_len = len(self.train_dataset)
        self._syn_len = len(cur_syn_dataset)
        self._timing_end(
            mixed_token, "mixed_dataset_build",
            real_len=int(self._real_len), syn_len=int(self._syn_len), total_len=int(total_len),
        )

        loader_token = self._timing_begin()
        self.local_cur_loaders = [
            DataLoader(
                mixed_ds, batch_size=self.args["local_bs0"], shuffle=True,
                num_workers=self._worker_count("base_mix_num_workers", 0), pin_memory=True,
            )
        ]
        self._timing_end(loader_token, "mixed_loader_build", batch_size=int(self.args["local_bs0"]))

        # ---- CLIP scores for the synthetic half of the mix ----
        collect_token = self._timing_begin()
        syn_labs = list(cur_syn_dataset.labels)
        self._timing_end(collect_token, "syn_tensor_collect", syn_items=int(len(syn_labs)))

        # Full-length buffers; the real prefix stays zero and is never read by
        # the mask regulariser because it is always multiplied by ``syn_flags``.
        sim_full = torch.zeros(total_len, dtype=torch.float32)
        div_full = torch.zeros(total_len, dtype=torch.float32)

        if syn_labs and preselected_sim is not None:
            syn_sim, syn_div = preselected_sim, preselected_div
            sim_full[self._real_len : self._real_len + self._syn_len] = syn_sim[: self._syn_len]
            div_full[self._real_len : self._real_len + self._syn_len] = syn_div[: self._syn_len]
        elif syn_labs:
            to_tensor = transforms.ToTensor()
            syn_tensors = [to_tensor(Image.fromarray(image)) for image in cur_syn_dataset.images]
            uniq = sorted(set(syn_labs))
            dataset_class_names = self._get_class_names()
            clip_token = self._timing_begin()
            syn_sim, syn_div = compute_clip_scores_memory(
                syn_tensors, syn_labs, [dataset_class_names[c] for c in uniq],
                device=self._device, templates=self._clip_templates(),
            )
            self._timing_end(
                clip_token, "clip_score_compute",
                syn_items=int(len(syn_labs)), syn_classes=int(len(uniq)),
            )
            # build_indexed_concat appends synthetic samples after the real ones.
            sim_full[self._real_len : self._real_len + self._syn_len] = syn_sim[: self._syn_len]
            div_full[self._real_len : self._real_len + self._syn_len] = syn_div[: self._syn_len]

        mask_token = self._timing_begin()
        self.sim_scores = sim_full.to(self._device)
        self.div_scores = div_full.to(self._device)
        self.mask_bank = MaskBank(
            num_samples=total_len,
            scale=float(self.args.get("mask_scale", 65.0)),
            device=self._device,
        )
        self._timing_end(mask_token, "mask_bank_build", total_len=int(total_len))

    def _build_protonet_loader_with_selected_syn(self, base_loader: DataLoader) -> DataLoader:
        """Append the rule's top-k synthetic images per new class to the support set."""
        n_per_class = int(self.args.get("protonet_syn_per_class", 0))
        if n_per_class <= 0:
            return base_loader

        cur_labels = list(range(self._known_classes, self._total_classes))
        to_tensor = transforms.ToTensor()

        cand_token = self._timing_begin()
        cand_tensors, cand_labels = [], []
        for lab in cur_labels:
            for p in self._get_current_task_syn_paths_for_class(lab):
                with Image.open(p) as im:
                    cand_tensors.append(to_tensor(im.convert("RGB")))
                cand_labels.append(int(lab))
        self._timing_end(
            cand_token, "protonet_candidate_load",
            candidate_count=int(len(cand_tensors)),
            candidate_classes=int(len(set(cand_labels))),
        )

        if not cand_tensors:
            return base_loader

        real_proto_dict = None
        if self.rule_mode == "3d":
            with self._timed("protonet_real_proto_compute"):
                real_proto_dict = self._compute_real_class_prototypes(base_loader)

        score_pack = self._score_rule_candidates(cand_tensors, cand_labels, real_proto_dict)
        q, m = score_pack["q"], score_pack["m"]

        idxs_by_lab = defaultdict(list)
        for i, lab in enumerate(cand_labels):
            idxs_by_lab[int(lab)].append(i)

        extras_by_label, extra_scores_by_label = {}, {}
        for lab in cur_labels:
            idxs = idxs_by_lab.get(int(lab), [])
            k = min(n_per_class, len(idxs))
            if k <= 0:
                continue
            topk_rel = torch.topk(q[idxs], k=k, largest=True).indices.tolist()
            chosen_idx = [idxs[j] for j in topk_rel]
            extras_by_label[int(lab)] = [cand_tensors[j] for j in chosen_idx]
            if self.rule_mode == "3d":
                # Only 3d mode reuses the rule's soft weights for prototype fusion.
                extra_scores_by_label[int(lab)] = [float(m[j].item()) for j in chosen_idx]

        if not any(len(v) > 0 for v in extras_by_label.values()):
            return base_loader

        aug_ds = AugmentedPerClassConcatDataset(
            base_ds=base_loader.dataset,
            extras_by_label=extras_by_label,
            extra_transform=self._make_protonet_extra_transform(
                getattr(base_loader.dataset, "transform", None)
            ),
            extra_scores_by_label=extra_scores_by_label if self.rule_mode == "3d" else None,
        )
        return DataLoader(
            aug_ds, batch_size=base_loader.batch_size, shuffle=False,
            num_workers=self._worker_count("protonet_num_workers", 0),
            pin_memory=getattr(base_loader, "pin_memory", False),
        )

    def _protonet_extra_transform_pil(self):
        """Transform applied to synthetic prototype candidates.

        NOTE (behaviour preserved from the original implementation): this path
        historically normalised with CIFAR-100 statistics for *every* dataset,
        while the real support images produced by ``utils/data.py`` are only
        ``ToTensor``-ed, i.e. left in ``[0, 1]``. Unifying the two pipelines
        changes published numbers, so the original convention is kept here and
        documented rather than silently altered.
        """
        mean, std = PROTONET_EXTRA_STATS
        return transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def _make_protonet_extra_transform(self, base_tf):
        """Apply the real support set's transform to in-memory synthetic tensors."""
        to_pil = transforms.ToPILImage()
        fallback = self._protonet_extra_transform_pil()

        def _tf(x):
            if isinstance(x, torch.Tensor):
                x = to_pil(x.detach().cpu().clamp(0.0, 1.0))
            if base_tf is not None:
                try:
                    return base_tf(x)
                except Exception:
                    return fallback(x)
            return fallback(x)

        return _tf

    # ------------------------------------------------------------------ #
    # mask logging
    # ------------------------------------------------------------------ #
    def _reset_mask_log(self):
        self._epoch_real_kept = 0.0
        self._epoch_syn_kept = 0.0
        self._epoch_real_total = 0
        self._epoch_syn_total = 0

    @torch.no_grad()
    def _accumulate_mask_log(self, gidx, mask_batch):
        if mask_batch is None or self._real_len + self._syn_len == 0:
            return
        real_flags = (gidx < self._real_len).float()
        syn_flags = 1.0 - real_flags

        self._epoch_real_total += int(real_flags.sum().item())
        self._epoch_real_kept += float(real_flags.sum().item())
        self._epoch_syn_total += int(syn_flags.sum().item())
        self._epoch_syn_kept += float((mask_batch * syn_flags).sum().item())

    def _log_mask_epoch(self, epoch_idx):
        if self._epoch_real_total + self._epoch_syn_total == 0:
            return
        real_ratio = self._epoch_real_kept / max(1, self._epoch_real_total)
        syn_ratio = self._epoch_syn_kept / max(1, self._epoch_syn_total)
        logging.info(
            "[mask][epoch %d] kept_real=%.1f/%d (%.3f) kept_syn=%.1f/%d (%.3f)",
            epoch_idx + 1,
            self._epoch_real_kept, self._epoch_real_total, real_ratio,
            self._epoch_syn_kept, self._epoch_syn_total, syn_ratio,
        )


class TaskSynImageDataset(Dataset):
    """Synthetic images of one task, capped at ``size_per_cls`` images per class.

    Images are decoded eagerly into memory because the base-session mix is
    iterated for every epoch and the set is small by construction.
    """

    def __init__(
        self, candidate_paths, size_per_cls, transform=None, selection_mode="head"
    ):
        self.transform = transform
        self.images, self.labels = [], []

        selection_mode = str(selection_mode).lower()
        if selection_mode not in {"head", "uniform"}:
            raise ValueError(
                "selection_mode must be 'head' or 'uniform'; got {!r}.".format(
                    selection_mode
                )
            )

        paths_by_class = defaultdict(list)
        for path in candidate_paths:
            try:
                label = int(os.path.basename(os.path.dirname(path)))
            except ValueError:
                continue
            paths_by_class[label].append(path)

        for label, class_paths in paths_by_class.items():
            keep = min(int(size_per_cls), len(class_paths))
            if selection_mode == "uniform":
                selected_indices = self._uniform_region_indices(len(class_paths), keep)
            else:
                selected_indices = list(range(keep))

            for index in selected_indices:
                path = class_paths[index]
                with Image.open(path) as img:
                    self.images.append(np.array(img.convert("RGB")))
                self.labels.append(label)

    @staticmethod
    def _uniform_region_indices(total, keep):
        """Select deterministic centre indices from ``keep`` equal regions."""
        total = int(total)
        keep = min(int(keep), total)
        if total <= 0 or keep <= 0:
            return []
        if keep == total:
            return list(range(total))
        return [int((region + 0.5) * total / keep) for region in range(keep)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        if self.transform is not None:
            img = self.transform(img)
        return idx, img, self.labels[idx]


class AugmentedPerClassConcatDataset(Dataset):
    """Real support set with per-class synthetic extras spliced in.

    Extras for a class are inserted right after that class' last real sample, so
    that ``shuffle=False`` prototype construction still sees contiguous classes.
    Exposes ``is_syn`` and ``sample_weight`` aligned with the returned indices,
    which is what :meth:`Learner.replace_fc` reads.
    """

    def __init__(self, base_ds, extras_by_label, extra_transform=None, extra_scores_by_label=None):
        self.base_ds = base_ds
        self.extras_by_label = extras_by_label
        self.extra_transform = extra_transform
        self.extra_scores_by_label = extra_scores_by_label or {}

        base_labels = getattr(base_ds, "labels", None)
        if base_labels is None:
            base_labels = [self._item_label(base_ds[i]) for i in range(len(base_ds))]
        elif isinstance(base_labels, torch.Tensor):
            base_labels = base_labels.cpu().tolist()
        self.base_labels = [int(x) for x in base_labels]

        idxs_by_label = defaultdict(list)
        for i, lab in enumerate(self.base_labels):
            idxs_by_label[lab].append(i)
        last_real_idx = {lab: idxs[-1] for lab, idxs in idxs_by_label.items()}

        self._order, self.labels, self.is_syn, self.sample_weight = [], [], [], []
        for i, lab in enumerate(self.base_labels):
            self._order.append(("base", i))
            self.labels.append(lab)
            self.is_syn.append(0)
            self.sample_weight.append(1.0)

            if i == last_real_idx[lab]:
                extras = self.extras_by_label.get(lab, [])
                scores = self.extra_scores_by_label.get(lab, [1.0] * len(extras))
                for k in range(len(extras)):
                    self._order.append(("extra", (lab, k)))
                    self.labels.append(lab)
                    self.is_syn.append(1)
                    self.sample_weight.append(float(scores[k]))

    @staticmethod
    def _item_label(item):
        return int(item[2] if isinstance(item, (tuple, list)) and len(item) >= 3 else item[1])

    def __len__(self):
        return len(self._order)

    def __getitem__(self, new_idx):
        src, payload = self._order[new_idx]
        if src == "base":
            item = self.base_ds[payload]
            if isinstance(item, (tuple, list)) and len(item) >= 3:
                _, img, lab = item[0], item[1], item[2]
            else:
                img, lab = item[0], item[1]
            return new_idx, img, int(lab)

        lab, k = payload
        img = self.extras_by_label[lab][k]
        if self.extra_transform is not None:
            img = self.extra_transform(img)
        return new_idx, img, int(lab)
