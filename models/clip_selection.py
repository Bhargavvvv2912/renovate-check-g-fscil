"""CLIP-based quality scoring for synthetic images.

The semantic, class-wise k-NN diversity, and mask-oriented scoring design is
adapted from https://github.com/Jackbrocp/clip-powered-data-selection for the
synthetic candidate pools used by G-FSCIL.

Two scalar scores are produced per synthetic sample and later consumed by the
learnable curation mask (base session) and by the transferable rule scorer
(incremental sessions):

* ``sim`` -- CLIP image/text alignment with the sample's own class name,
  min-max normalised over the batch of candidates.
* ``div`` -- class-wise k-NN mean distance in CLIP image-feature space, min-max
  normalised. A higher value means the sample sits in a sparser region of its
  class, i.e. it contributes more diversity.

Both the CLIP backbone and the text embeddings are cached process-wide so that
repeated scoring across tasks never re-runs the text encoder.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

import clip

from utils.prompts import DEFAULT_CLIP_TEMPLATES

CLIP_MODEL_NAME = "ViT-B/16"

_CLIP_MODEL_CACHE = {}
_CLIP_TEXT_CACHE = {}

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def get_clip_model(model_name=CLIP_MODEL_NAME, device=None):
    """Return one process-wide CLIP instance shared by scoring and calibration."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    key = (model_name, str(device))
    if key not in _CLIP_MODEL_CACHE:
        model, _ = clip.load(model_name, device=device)
        _CLIP_MODEL_CACHE[key] = model.float().eval()
    return _CLIP_MODEL_CACHE[key]


@torch.no_grad()
def build_clip_text_features(model, model_name, class_names, device, templates=None):
    """Encode class names with an optional prompt ensemble and cache the result."""
    templates = tuple(templates or DEFAULT_CLIP_TEMPLATES)
    key = (model_name, str(device), tuple(str(x) for x in class_names), templates)
    if key not in _CLIP_TEXT_CACHE:
        per_template = []
        clean_names = [str(name).replace("_", " ") for name in class_names]
        for template in templates:
            text_inputs = clip.tokenize([template.format(name) for name in clean_names]).to(device)
            features = F.normalize(model.encode_text(text_inputs), dim=-1)
            per_template.append(features)
        text_features = F.normalize(torch.stack(per_template, dim=0).mean(dim=0), dim=-1)
        _CLIP_TEXT_CACHE[key] = text_features.detach()
    return _CLIP_TEXT_CACHE[key]


def _resize_chw_tensor_to(t: torch.Tensor, size: int) -> torch.Tensor:
    """Coerce a tensor to ``[3, size, size]`` on CPU float32.

    Accepts ``[C, H, W]`` or ``[1, C, H, W]``; grayscale input is broadcast to
    three channels.
    """
    if t.dim() == 3:
        t4 = t.unsqueeze(0)
    elif t.dim() == 4 and t.size(0) == 1:
        t4 = t
    else:
        raise ValueError("Unsupported tensor shape: {}".format(tuple(t.shape)))

    if t4.size(1) == 1:
        t4 = t4.repeat(1, 3, 1, 1)
    elif t4.size(1) != 3:
        raise ValueError("Expected 1 or 3 channels, got C={}".format(t4.size(1)))

    if t4.shape[-2:] != (size, size):
        t4 = F.interpolate(t4.float(), size=(size, size), mode="bilinear", align_corners=False)

    return t4.squeeze(0).contiguous()


def _prepare_clip_batch(image_tensors, size):
    fixed = []
    for tensor in image_tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("compute_clip_scores_memory expects Tensor, got {}".format(type(tensor)))
        tensor = _resize_chw_tensor_to(tensor.detach().cpu(), size).float()
        if float(tensor.min()) < -1e-5 or float(tensor.max()) > 1.00001:
            raise ValueError(
                "CLIP scoring expects image tensors in [0, 1]. "
                "Pass raw ToTensor() images, not dataset-normalised tensors."
            )
        fixed.append(tensor)

    batch = torch.stack(fixed, dim=0)
    mean = torch.tensor(CLIP_MEAN, dtype=batch.dtype).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, dtype=batch.dtype).view(1, 3, 1, 1)
    return (batch - mean) / std


def _clip_input_resolution(model, default=224):
    try:
        return int(model.visual.input_resolution)
    except Exception:
        return default


@torch.no_grad()
def compute_clip_scores_memory(image_tensors, image_labels, class_names, device=None,
                               batch_size=32, templates=None):
    """Compute ``(sim_scores, div_scores)`` for in-memory image tensors.

    Args:
        image_tensors: list of ``Tensor[C, H, W]`` in ``[0, 1]``; sizes may differ.
        image_labels: list of int labels, one per tensor.
        class_names: text names for ``sorted(set(image_labels))``, in that order.
        device: torch device used for the CLIP forward passes.
        batch_size: CLIP forward batch size.

    Returns:
        Two CPU ``Tensor[N]`` in ``[0, 1]``: ``(sim_scores, div_scores)``.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = get_clip_model(CLIP_MODEL_NAME, device)
    target_res = _clip_input_resolution(model)

    imgs_tensor = _prepare_clip_batch(image_tensors, target_res).to(device)  # [N, 3, R, R]

    unique_labels = sorted(set(int(x) for x in image_labels))
    if len(class_names) < len(unique_labels):
        raise ValueError(
            "class_names has {} entries but {} distinct labels were given.".format(
                len(class_names), len(unique_labels)
            )
        )
    label_to_col = {lab: i for i, lab in enumerate(unique_labels)}
    mapped_labels = torch.tensor(
        [label_to_col[int(l)] for l in image_labels], device=device, dtype=torch.long
    )

    text_features = build_clip_text_features(
        model, CLIP_MODEL_NAME, list(class_names)[: len(unique_labels)], device, templates
    )

    all_feats, all_scores = [], []
    for start in range(0, imgs_tensor.size(0), batch_size):
        end = start + batch_size
        feats = model.encode_image(imgs_tensor[start:end])
        feats = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats)

        sim = feats @ text_features.T
        labs = mapped_labels[start:end]
        all_scores.append(torch.gather(sim, 1, labs.unsqueeze(1)).squeeze(1))

    all_feats = torch.cat(all_feats, dim=0)     # [N, D]
    sim_scores = torch.cat(all_scores, dim=0)   # [N]
    sim_scores = (sim_scores - sim_scores.min()) / (sim_scores.max() - sim_scores.min() + 1e-8)

    # Diversity: class-wise k-NN mean distance in CLIP feature space.
    div_scores = torch.zeros(all_feats.size(0), device=device, dtype=torch.float32)
    feats_np = all_feats.detach().cpu().numpy()
    labs_np = mapped_labels.detach().cpu().numpy()

    for cls in np.unique(labs_np):
        idxs = np.where(labs_np == cls)[0]
        sub = feats_np[idxs]
        k = min(50, sub.shape[0])
        if k > 1:
            nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(sub)
            dists, _ = nbrs.kneighbors(sub)
            div_vals = dists[:, 1:].mean(axis=1).astype(np.float32)
        else:
            div_vals = np.zeros(sub.shape[0], dtype=np.float32)
        div_scores[idxs] = torch.from_numpy(div_vals).to(device)

    div_scores = (div_scores - div_scores.min()) / (div_scores.max() - div_scores.min() + 1e-8)

    return sim_scores.detach().cpu(), div_scores.detach().cpu()
