"""Dataset feeding the textual-inversion stage of the latent diffusion model.

The class-inversion data contract follows the DDDR generation pipeline at
https://github.com/jinglin-liang/DDDR and is extended here to path-backed CUB
and ImageNet-R datasets.

Each item is a dict shaped the way ``LatentDiffusion.shared_step`` expects:

``image``          -- ``HWC`` float32 array in ``[-1, 1]``
``caption``        -- a random template containing the placeholder token
``abs_class_id``   -- the mapped (global) class id
``rel_class_id``   -- the class id relative to the first class of this task,
                      which is what ``EmbeddingManager`` indexes with

The dataset accepts either an array of decoded images (CIFAR-style datasets) or
an array of file paths (CUB / ImageNet-R style ``ImageFolder`` datasets), so the
same inversion code runs on every benchmark.
"""

import random

import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.prompts import imagenetr_prefix_prompt

INTERPOLATIONS = {
    "bilinear": PIL.Image.BILINEAR,
    "bicubic": PIL.Image.BICUBIC,
    "lanczos": PIL.Image.LANCZOS,
}

# Keep this list byte-for-byte equivalent to the original inversion policy.
# Repeated strings are intentional sampling weights, not redundant entries.
IMAGENET_TEMPLATES = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
    "an illustration of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "an illustration of a clean {}",
    "an illustration of a dirty {}",
    "a dark photo of the {}",
    "an illustration of my {}",
    "an illustration of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "an illustration of the {}",
    "a good photo of the {}",
    "an illustration of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "an illustration of the clean {}",
    "a rendition of a {}",
    "an illustration of a nice {}",
    "a good photo of a {}",
    "an illustration of the nice {}",
    "an illustration of the small {}",
    "an illustration of the weird {}",
    "an illustration of the large {}",
    "an illustration of a cool {}",
    "an illustration of a small {}",
    "a depiction of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a depiction of a clean {}",
    "a depiction of a dirty {}",
    "a dark photo of the {}",
    "a depiction of my {}",
    "a depiction of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a depiction of the {}",
    "a good photo of the {}",
    "a depiction of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a depiction of the clean {}",
    "a rendition of a {}",
    "a depiction of a nice {}",
    "a good photo of a {}",
    "a depiction of the nice {}",
    "a depiction of the small {}",
    "a depiction of the weird {}",
    "a depiction of the large {}",
    "a depiction of a cool {}",
    "a depiction of a small {}",
]


class GenDataset(Dataset):
    def __init__(self,
                 images,
                 class_ids,
                 use_path=False,
                 min_class_id=0,
                 size=256,
                 repeats=100,
                 interpolation="bicubic",
                 flip_p=0.5,
                 placeholder_token="*",
                 center_crop=False,
                 class_names=None,
                 dataset_name="cifar224",
                 syn_prompt_mode="default",
                 syn_prompt_fallback_template="a rendition of *"):
        """
        Args:
            images: decoded ``HWC uint8`` arrays, or file paths when ``use_path``.
            class_ids: mapped class id per image.
            use_path: set ``True`` for ``ImageFolder``-backed datasets.
            min_class_id: first class id of the current task.
            repeats: virtual epoch multiplier, so short k-shot sets still yield
                enough inversion steps.
            class_names: readable names indexed by mapped/global class id.
            dataset_name: selects the ImageNet-R prefix-aware caption policy.
        """
        self.images = images
        self.class_ids = class_ids
        self.use_path = use_path
        self.min_class_id = min_class_id

        self.num_images = len(self.images)
        self._length = self.num_images * repeats

        self.placeholder_token = placeholder_token
        self.class_names = class_names
        self.dataset_name = str(dataset_name).lower()
        self.syn_prompt_mode = str(syn_prompt_mode).lower()
        self.syn_prompt_fallback_template = str(syn_prompt_fallback_template)
        self.center_crop = center_crop
        self.size = size
        self.interpolation = INTERPOLATIONS[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def _load(self, i):
        if self.use_path:
            with Image.open(self.images[i]) as img:
                return img.convert("RGB")
        return Image.fromarray(self.images[i]).convert("RGB")

    def _class_name(self, class_id):
        if isinstance(self.class_names, dict):
            return self.class_names.get(int(class_id))
        if isinstance(self.class_names, (list, tuple)) and 0 <= int(class_id) < len(self.class_names):
            return self.class_names[int(class_id)]
        return None

    def _caption(self, i, class_id):
        if self.dataset_name == "imagenetr" and self.syn_prompt_mode == "prefix":
            path = self.images[i] if self.use_path else ""
            prompt = imagenetr_prefix_prompt(
                path, self._class_name(class_id), self.placeholder_token
            )
            if prompt is not None:
                return prompt
            return self.syn_prompt_fallback_template
        return random.choice(IMAGENET_TEMPLATES).format(self.placeholder_token)

    def __getitem__(self, i):
        i = i % self.num_images
        image = self._load(i)
        class_id = int(self.class_ids[i])

        example = {
            "abs_class_id": class_id,
            "rel_class_id": class_id - self.min_class_id,
            "caption": self._caption(i, class_id),
        }

        img = np.array(image).astype(np.uint8)
        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)
        image = self.flip(image)

        example["image"] = (np.array(image).astype(np.uint8) / 127.5 - 1.0).astype(np.float32)
        return example
