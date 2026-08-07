<div align="center">

# Learning to Curate What You Generate for Generalizable Few-Shot Class-Incremental Learning

### Official PyTorch Implementation · ACM Multimedia 2026

**Junhui Yin<sup>\*</sup>, Yuchen Yang<sup>\*</sup>, Yilin Yin, Shuai Na, Haoran Xi, Jianhua Yang<sup>†</sup>, Muyi Sun, Man Zhang, Shengfeng He**

<sup>\*</sup> Equal contribution &nbsp;&nbsp; <sup>†</sup> Corresponding author

[![Conference](https://img.shields.io/badge/ACM%20Multimedia-2026-7b2cbf.svg)](https://2026.acmmm.org/)
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3767308.3836063-00629B.svg)](https://doi.org/10.1145/3767308.3836063)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Paper](https://doi.org/10.1145/3767308.3836063) ·
[Code](https://github.com/NiHaoWoJiaoYYC/G-FSCIL) ·
[Results](#latest-results) ·
[Models](#pretrained-models) ·
[Generated Images](#pre-generated-synthetic-images) ·
[Citation](#citation)

</div>

---

## Overview

This repository contains the official implementation of **Learning to Curate
What You Generate for Generalizable Few-Shot Class-Incremental Learning**. We
study Generalizable Few-Shot Class-Incremental Learning (G-FSCIL), in which the
base session contains only a small number of classes and each subsequent
session introduces new classes with limited supervision.

The method combines two complementary components:

- **Trustworthy Synthetic Knowledge Curation (TSKC)** constructs a
  class-specific synthetic candidate pool, learns a semantic- and
  diversity-aware selection mask in the base session, and distils its behavior
  into a transferable policy scorer for later sessions.
- **Boundary-Stable Incremental Adaptation (BSIA)** augments new-class
  prototypes with curated synthetic knowledge and calibrates confusable
  old/new decision boundaries while updating only the prompt encoder.

Class inversion and image generation are performed once when a class is first
observed. The resulting class condition and candidate pool are cached and can
be reused across later runs, seeds, and ablations.

```mermaid
flowchart LR
    A["Few-shot real samples"] --> B["Class inversion"]
    B --> C["Synthetic candidate pool"]
    C --> D["Semantic and diversity curation"]
    D --> E["Transferable policy scorer"]
    E --> F["Prototype enhancement"]
    F --> G["Boundary-stable adaptation"]
```

## G-FSCIL Protocols

The release provides the three experimental settings used by this codebase.
Each JSON file is self-contained and records the complete class order and
hyperparameter set.

| Configuration | Dataset | Base session | Incremental protocol | Total sessions |
| --- | --- | ---: | --- | ---: |
| [`exps/cifar.json`](exps/cifar.json) | CIFAR-100 | 5 classes | 19 × 5-way 5-shot | 20 |
| [`exps/cub.json`](exps/cub.json) | CUB-200-2011 | 10 classes | 19 × 10-way 5-shot | 20 |
| [`exps/inr.json`](exps/inr.json) | ImageNet-R | 10 classes | 19 × 10-way 5-shot | 20 |

Internal identifiers such as `asp`, `model_prefix`, and the original flat
configuration names are retained for compatibility with existing experiment
records. They refer to the ASP implementation basis and do not change the paper
or repository name.

## Latest Results

The table reports our latest best complete runs under the 20-session G-FSCIL
protocol. All values are percentages from a single run with seed 2024. **Avg.
Acc.** is the mean top-1 accuracy across all sessions; the remaining final
metrics are evaluated after session 19.

| Dataset | Base Acc. | Avg. Acc. | Final Acc. | Final HAcc | Final Old Acc. | Final New Acc. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CUB-200-2011 | 97.12 | **81.91** | 78.01 | **82.30** | 87.65 | 77.57 |
| ImageNet-R | 89.71 | **73.09** | 70.05 | **72.29** | 74.41 | 70.29 |

<details>
<summary><b>Session-wise top-1 accuracy curves</b></summary>

- **CUB-200-2011:** 97.12, 93.98, 84.89, 84.77, 84.67, 83.33, 79.96,
  80.72, 81.91, 82.61, 81.29, 80.23, 78.89, 78.65, 77.42, 77.13, 77.78,
  77.01, 77.83, 78.01.
- **ImageNet-R:** 89.71, 80.50, 75.81, 75.24, 74.63, 75.14, 76.21,
  72.33, 71.88, 71.70, 69.93, 70.02, 68.99, 69.67, 69.91, 70.12, 69.68,
  70.17, 70.01, 70.05.

</details>

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/NiHaoWoJiaoYYC/G-FSCIL.git
cd G-FSCIL
```

### 2. Create the environment

Python 3.10 is recommended. The reference dependency set combines the prompted
ViT backbone, CLIP-based curation, and latent-diffusion generation in one
environment.

```bash
conda create -n gfscil python=3.10 -y
conda activate gfscil

python -m pip install --upgrade pip
python -m pip install setuptools==69.5.1 wheel==0.43.0
```

Install the PyTorch build matching your CUDA runtime. The reference setup uses
CUDA 12.8:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

For a different CUDA version, select the corresponding wheel from the
[official PyTorch installer](https://pytorch.org/get-started/locally/) before
installing `requirements.txt`. CPU and macOS installations are suitable for
configuration and data-pipeline checks, but full training and synthetic image
generation are designed for NVIDIA GPUs.

Verify the main dependencies:

```bash
python -c "import torch, torchvision, timm, clip, sklearn, omegaconf, pytorch_lightning, transformers; print('torch:', torch.__version__, '| torchvision:', torchvision.__version__, '| cuda:', torch.cuda.is_available())"
```

<details>
<summary><b>Why is setuptools pinned?</b></summary>

OpenAI CLIP and parts of the legacy latent-diffusion dependency chain still
import `pkg_resources`. Recent setuptools releases may no longer provide it in
the expected form. If an existing environment raises
`ModuleNotFoundError: No module named 'pkg_resources'`, repair it with:

```bash
python -m pip install --force-reinstall setuptools==69.5.1 wheel==0.43.0
python -c "import pkg_resources; print('pkg_resources: OK')"
```

</details>

### 3. Prepare models, datasets, and synthetic images

Before training, prepare the assets described in the following three sections:

1. [Pretrained models](#pretrained-models)
2. [Datasets](#datasets)
3. [Pre-generated synthetic images](#pre-generated-synthetic-images), or
   enable generation from scratch

### 4. Validate a configuration

Configuration validation does not import the heavy training stack:

```bash
python main.py --config exps/cifar.json --check-config
python main.py --config exps/cub.json --check-config
python main.py --config exps/inr.json --check-config
```

To inspect the fully resolved configuration:

```bash
python main.py --config exps/cifar.json --print-config --check-config
```

### 5. Start training

Pass a physical GPU ID to the corresponding launcher:

```bash
bash scripts/run_cifar.sh 0
bash scripts/run_cub.sh 0
bash scripts/run_inr.sh 0
```

See [Training and generation](#training-and-generation) for the complete
cached-image and generate-from-scratch workflows.

## Environment Details

| Component | Reference version | Role |
| --- | ---: | --- |
| Python | 3.10 | Runtime |
| PyTorch | 2.7.1 | Training and generation |
| torchvision | 0.22.1 | Data loading and augmentation |
| timm | 1.0.19 | ImageNet-pretrained ViT-B/16 |
| OpenAI CLIP | 1.0.1 | Semantic scoring and calibration |
| PyTorch Lightning | 2.5.5 | Latent-diffusion compatibility |
| transformers | 4.55.0 | Local BERT tokenizer |

The upstream implementations were released with different dependency stacks.
`requirements.txt` provides one tested package set for the integrated pipeline;
do not concatenate the original environment files from ASP, DDDR, and the CLIP
selection repository.

## Pretrained Models

Two assets are required only when generating synthetic images from scratch.
The ViT and CLIP weights are downloaded automatically on first use.

| Asset | Purpose | Download | Expected location |
| --- | --- | --- | --- |
| LDM text-to-image checkpoint | Class inversion and DDIM synthesis | [Download checkpoint](https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt) | `models/ldm/text2img-large/model.ckpt` |
| BERT base uncased tokenizer | LDM text conditioning | [Hugging Face](https://huggingface.co/google-bert/bert-base-uncased) | `models/bert/` |
| ImageNet-pretrained ViT-B/16 | FSCIL backbone | Downloaded automatically by [`timm`](https://github.com/huggingface/pytorch-image-models) | Framework cache |
| OpenAI CLIP ViT-B/16 | Synthetic curation and semantic calibration | Downloaded automatically by [OpenAI CLIP](https://github.com/openai/CLIP) | `~/.cache/clip/` by default |

Download the LDM checkpoint:

```bash
mkdir -p models/ldm/text2img-large
curl -L \
  https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt \
  -o models/ldm/text2img-large/model.ckpt
```

Create the local BERT tokenizer directory without downloading unnecessary BERT
model weights:

```bash
python -c "from transformers import BertTokenizerFast; BertTokenizerFast.from_pretrained('google-bert/bert-base-uncased').save_pretrained('models/bert')"
```

The resulting layout should be:

```text
models/
├── bert/
│   ├── special_tokens_map.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── vocab.txt
└── ldm/
    └── text2img-large/
        └── model.ckpt
```

You can prefetch the automatically managed ViT and CLIP weights with:

```bash
python -c "import timm; timm.create_model('vit_base_patch16_224', pretrained=True); print('ViT-B/16: ready')"
python -c "import clip; clip.load('ViT-B/16', device='cpu'); print('CLIP ViT-B/16: ready')"
```

If `need_syn_imgs` is `false` and a complete synthetic cache is available, the
LDM checkpoint and local BERT tokenizer are not loaded. The standard launchers
train the full G-FSCIL sequence from the ViT initialization; base-session
checkpoints are written automatically to `saved_model/` during execution.

## Datasets

All default paths are relative to `./data`. CIFAR-100 is downloaded
automatically. CUB-200-2011 and ImageNet-R must be prepared as `ImageFolder`
train/test splits.

| Dataset | Source | Expected root |
| --- | --- | --- |
| CIFAR-100 | [Official dataset](https://www.cs.toronto.edu/~kriz/cifar.html), downloaded by torchvision | `data/cifar-100-python/` |
| CUB-200-2011 | [Prepared split](https://drive.google.com/file/d/1jx0ICqvgaXyfWUVLTv6St0_b6p7D0Hpm/view?usp=sharing) · [Dataset page](https://www.vision.caltech.edu/datasets/cub_200_2011/) | `data/cub/` |
| ImageNet-R | [Prepared split](https://drive.google.com/file/d/1R4bRjYXnbRWje6hw_YPdsKr1HuojXzqO/view?usp=sharing) · [Official project](https://github.com/hendrycks/imagenet-r) | `data/imagenet-r/` |

Expected directory structure:

```text
data/
├── cifar-100-python/                  # created automatically
├── cub/
│   ├── train/
│   │   └── <class-name>/*.jpg
│   └── test/
│       └── <class-name>/*.jpg
└── imagenet-r/
    ├── classnames.txt
    ├── train/
    │   └── <wnid>/*.jpg
    └── test/
        └── <wnid>/*.jpg
```

For CUB and ImageNet-R, the class-folder names and ordering must match between
`train/` and `test/`. ImageNet-R must contain the same 200 WNID folders in both
splits. Its optional `classnames.txt` records one WNID/name pair per line:

```text
n01443537 goldfish
n01484850 great white shark
n01494475 hammerhead shark
```

The repository contains a fallback list, but an explicit file is recommended
for an auditable WNID-to-name mapping. Check the layout before a long run:

```bash
python tools/check_imagenetr.py --root ./data/imagenet-r
```

## Pre-generated Synthetic Images

> **Download:** [Pre-generated synthetic images (Google Drive)](https://drive.google.com/file/d/1GRn_UjeJTHfrmlNGnK8gBinvoid5HAS6/view?usp=drive_link)

Using the released synthetic candidates is the recommended route for fast and
deterministic reproduction. The archive contains the top-level `outputs/`
folder. Extract it directly into the repository root so that the three cache
roots exactly match the paths already stored in the public JSON files. Avoid
creating a nested `outputs/outputs/` directory.

| Dataset | Configuration key | Required cache root | Expected images |
| --- | --- | --- | ---: |
| CIFAR-100 | `exps/cifar.json` | `outputs/syn_image_5_5_pre0.5_shot20_bs5/` | 20,000 |
| CUB-200-2011 | `exps/cub.json` | `outputs/base10/` | 40,000 |
| ImageNet-R | `exps/inr.json` | `outputs/syn_imagenetr_base10/` | 40,000 |

Each class has 200 generated candidates. The cache convention is:

```text
<syn_image_path>/
└── task_<task-id>/
    └── <class-id>/
        ├── <class-id>-0.jpg
        ├── <class-id>-1.jpg
        └── ...
```

After extraction, the relevant part of the repository should look like:

```text
outputs/
├── syn_image_5_5_pre0.5_shot20_bs5/  # CIFAR-100
├── base10/                            # CUB-200-2011
└── syn_imagenetr_base10/              # ImageNet-R
```

The downloaded archive must contain real, decodable JPEG images. Empty
directory stubs or zero-byte `.jpg` placeholders are not a usable cache. A
quick count of non-empty files is:

```bash
find outputs/syn_image_5_5_pre0.5_shot20_bs5 -type f -name '*.jpg' ! -size 0 | wc -l
find outputs/base10 -type f -name '*.jpg' ! -size 0 | wc -l
find outputs/syn_imagenetr_base10 -type f -name '*.jpg' ! -size 0 | wc -l
```

The expected counts are 20,000, 40,000, and 40,000, respectively. If the
archive uses different outer folder names, either rename them to the paths
above or update `syn_image_path` in the corresponding JSON.

## Training and Generation

### Route A: train with pre-generated images

This is the recommended reproduction workflow.

1. Prepare the real dataset under `data/`.
2. Extract the archive in the repository root, producing the `outputs/` paths
   listed above.
3. Keep `need_syn_imgs` set to `false` in the selected experiment JSON.
4. Validate the configuration and launch the experiment.

```bash
python main.py --config exps/cifar.json --check-config
bash scripts/run_cifar.sh 0
```

Repeat with `run_cub.sh` or `run_inr.sh` for the other datasets. Setting
`need_syn_imgs: false` disables new generation but does **not** disable
synthetic replay, prototype enhancement, or boundary calibration.

### Route B: generate once, then reuse

Use this route when the released synthetic archive is unavailable or when you
want to regenerate candidates.

1. Download the LDM checkpoint and BERT tokenizer.
2. Set `need_syn_imgs` to `true` in the selected JSON.
3. Confirm that `ldm_ckpt`, `config`, and `syn_image_path` point to valid paths.
4. Launch the same training command.

For example, after editing `exps/cifar.json`:

```bash
bash scripts/run_cifar.sh 0
```

No separate generation script is required. At each session, the program first
learns class-specific condition embeddings, generates 200 candidates for every
new class, writes them to the task cache, and then continues training. Once a
complete cache has been created, set `need_syn_imgs` back to `false` for later
seeds and ablations.

### GPU selection

The shell wrappers set `CUDA_VISIBLE_DEVICES`, while each JSON uses the
process-local logical device `"0"`:

```bash
bash scripts/run_inr.sh 3
```

This command exposes physical GPU 3 as `cuda:0`, so `device` should remain
`["0"]` in the JSON. You can also launch directly:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config exps/inr.json
```

## Configuration Guide

The three public JSON files are standalone. The most important path and mode
settings are:

| Key | Meaning |
| --- | --- |
| `data_root` | Root shared by CIFAR-100 and CUB-200-2011 |
| `imagenetr_root` | ImageNet-R root; overrides `data_root` for ImageNet-R |
| `imagenetr_classnames_txt` | Optional WNID-to-class-name mapping |
| `ldm_ckpt` | LDM checkpoint used when generation is enabled |
| `syn_image_path` | Dataset-specific synthetic candidate cache |
| `need_syn_imgs` | Generate missing candidates in the current run |
| `save_dir` | Experiment logs and metric files |
| `seed` | List of random seeds executed sequentially |
| `device` | Process-local CUDA device IDs |

Other parameter groups include:

- **Protocol:** `init_cls`, `increment`, `kshot`, `class_order`.
- **Prompt learning:** `prompt_token_num`, `vpt_type`, `avg_alpha`,
  `perturb_var`, `EMA_beta`, `TIP_init`, `anchor_lambda`, `kl_weight`.
- **Optimization:** `tuned_epoch`, `fs_epoch`, `init_lr`, `fs_lr`,
  `batch_size`, `fs_batch_size`, `weight_decay`, `optimizer`.
- **Generation:** `com_round_gen`, `g_local_train_steps`, `g_local_bs`,
  `n_iter`, `synthesis_n_samples`, `synthesis_guidance_scale`, `ddim_steps`.
- **Curation and policy:** `cur_size`, `clip_selection_ratio`,
  `clip_diversity_lambda`, `mask_scale`, `mask_lr_mul`, `mask_reg_weight`,
  `mask_reg_beta`, `rule_mode`, `rule_tau`, `rule_distill_weight`.
- **Prototype and calibration:** `protonet_syn_per_class`,
  `protonet_syn_topk`, `protonet_syn_ratio`, `calib_lambda`, `calib_m0`,
  `calib_m1`, `calib_sem_tau`, `calib_vis_tau`, `calib_fwd_weight`,
  `calib_rev_weight`.

## Outputs

A normal run writes logs, metrics, and the base-session checkpoint to:

```text
outputs/
├── exp/<log_name>.log
└── metrics/
    ├── perf_<prefix>_seed<seed>.jsonl
    └── timing_<prefix>_seed<seed>.jsonl

saved_model/
└── asp/<dataset>/<init>_<increment>/
    └── asp_<base-epochs>_<seed>_pretrained_vit_b16_224_vpt.pth
```

Performance records contain per-session top-1 accuracy, average accuracy,
harmonic accuracy, old-class accuracy, and new-class accuracy. Timing records
are produced when `timing_enabled` is set to `true`.

## Tests

Run the lightweight configuration and data-pipeline tests before a full
experiment:

```bash
python tests/test_config.py
python tests/test_data_pipeline.py
```

or:

```bash
python -m pytest -q tests/
```

The smoke tests use temporary synthetic inputs and do not claim end-to-end
numerical reproduction. Full experiments require the real datasets, pretrained
weights, generated candidate pools, and a CUDA environment.

## Troubleshooting

<details>
<summary><b>Synthetic replay is enabled, but cached images are missing</b></summary>

Check that `syn_image_path` points to the dataset-specific root and that every
task/class directory contains real JPEG images. Extract the released archive or
set `need_syn_imgs` to `true` after preparing the LDM and BERT assets.

</details>

<details>
<summary><b>PIL cannot identify a synthetic image</b></summary>

The cache contains an empty or corrupted `.jpg` file. Replace placeholder files
with the downloaded archive, regenerate the affected task, and rerun the
non-empty-file checks in the synthetic-image section.

</details>

<details>
<summary><b>CUDA device index is invalid</b></summary>

When a launcher exposes one physical GPU through `CUDA_VISIBLE_DEVICES`, keep
`device: ["0"]` in the JSON. The value is a logical index inside the process,
not the original physical GPU number.

</details>

<details>
<summary><b>ViT or CLIP download fails on the first run</b></summary>

The backbone and CLIP weights are fetched automatically. Ensure that the
machine has network access or prefetch the models with the commands in the
pretrained-model section. Subsequent runs use the local framework caches.

</details>

## Repository Structure

```text
G-FSCIL/
├── main.py                     # Configuration, validation, and entry point
├── trainer.py                  # Seed and incremental-session loops
├── exps/                       # CIFAR/CUB/ImageNet-R experiment JSON files
├── scripts/                    # GPU-aware launchers
├── models/
│   ├── asp.py                  # Integrated G-FSCIL learner
│   └── clip_selection.py       # CLIP semantic and diversity scoring
├── backbone/                   # Prompted ViT and cosine classifier
├── ldm/                        # Latent-diffusion generation modules
├── utils/                      # Data, configuration, prompts, and metrics
├── tools/                      # Dataset and timing utilities
├── tests/                      # CPU-friendly smoke tests
├── outputs/                    # Logs, metrics, and synthetic caches
└── saved_model/                # Base-session checkpoints
```

## Implementation Basis and Acknowledgements

This implementation integrates and adapts components from the following open
research projects:

| Component in this repository | Implementation basis |
| --- | --- |
| FSCIL framework and prompted ViT | [FSCIL-ASP](https://github.com/DawnLIU35/FSCIL-ASP) |
| Class inversion, latent diffusion, DDIM synthesis, and taskwise caches | [DDDR](https://github.com/jinglin-liang/DDDR) and [latent-diffusion](https://github.com/CompVis/latent-diffusion) |
| CLIP semantic scoring, class-wise diversity, and ratio-constrained masks | [CLIP-powered data selection](https://github.com/Jackbrocp/clip-powered-data-selection) and [OpenAI CLIP](https://github.com/openai/CLIP) |

The federated components of DDDR are not used. See [`NOTICE.md`](NOTICE.md) for
the corresponding repositories and papers. Upstream code, pretrained models,
and datasets remain subject to their original licenses and terms.

## Citation

If this work is useful in your research, please cite:

```bibtex
@inproceedings{yin2026learning,
  author    = {Yin, Junhui and Yang, Yuchen and Yin, Yilin and Na, Shuai and
               Xi, Haoran and Yang, Jianhua and Sun, Muyi and Zhang, Man and
               He, Shengfeng},
  title     = {Learning to Curate What You Generate for Generalizable Few-Shot
               Class-Incremental Learning},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
  year      = {2026},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  location  = {Rio de Janeiro, Brazil},
  doi       = {10.1145/3767308.3836063},
  url       = {https://doi.org/10.1145/3767308.3836063}
}
```

Please also cite FSCIL-ASP, DDDR, latent diffusion, OpenAI CLIP, and the
CLIP-powered data-selection work when reusing their corresponding components.

## License

This repository is released under the [MIT License](LICENSE). See
[`NOTICE.md`](NOTICE.md) for attribution and third-party information. Datasets,
pretrained models, and upstream components are distributed under their own
licenses.
