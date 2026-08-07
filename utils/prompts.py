"""Text-prompt policies shared by diffusion generation and CLIP scoring.

ImageNet-R is not a photo-only benchmark.  Its filenames commonly retain a
rendition/source prefix (for example ``art_3.jpg`` or ``origami_12.jpg``).  The
helpers below preserve that signal during textual inversion and synthesis, and
use a rendition-aware prompt ensemble for CLIP.  CIFAR and CUB keep the
original single-photo CLIP prompt so their behaviour is unchanged.
"""

import os
import re
from collections import Counter


DEFAULT_CLIP_TEMPLATES = ("a photo of a {}.",)

IMAGENETR_CLIP_TEMPLATES = (
    "a photo of a {}.",
    "a rendition of a {}.",
    "a drawing of a {}.",
    "a painting of a {}.",
    "a cartoon of a {}.",
    "a sketch of a {}.",
    "an illustration of a {}.",
    "a sculpture of a {}.",
    "a toy {}.",
    "a plushie {}.",
    "a video game {}.",
    "an origami {}.",
)


# Prefixes are normalised to a small set of prompt families.  The aliases are
# deliberately permissive because different ImageNet-R mirrors use slightly
# different spellings in their filenames.
IMAGENETR_PREFIX_ALIASES = {
    "art": "artwork",
    "artwork": "artwork",
    "cartoon": "cartoon",
    "deviantart": "digital_art",
    "digital_art": "digital_art",
    "drawing": "drawing",
    "embroidery": "embroidery",
    "embroidered": "embroidery",
    "graffiti": "graffiti",
    "graphic": "graphic",
    "illustration": "illustration",
    "mosaic": "mosaic",
    "origami": "origami",
    "painting": "painting",
    "pattern": "pattern",
    "plastic": "plastic",
    "plastic_object": "plastic",
    "plush": "plushie",
    "plushie": "plushie",
    "rendering": "rendition",
    "rendition": "rendition",
    "sculpture": "sculpture",
    "sketch": "sketch",
    "sticker": "sticker",
    "tattoo": "tattoo",
    "toy": "toy",
    "video_game": "video_game",
    "videogame": "video_game",
}

IMAGENETR_PREFIX_PROMPT_TEMPLATES = {
    "artwork": "an artwork of a {}",
    "cartoon": "a cartoon of a {}",
    "digital_art": "digital art of a {}",
    "drawing": "a drawing of a {}",
    "embroidery": "an embroidered {}",
    "graffiti": "graffiti of a {}",
    "graphic": "a graphic illustration of a {}",
    "illustration": "an illustration of a {}",
    "mosaic": "a mosaic of a {}",
    "origami": "an origami {}",
    "painting": "a painting of a {}",
    "pattern": "a pattern depicting a {}",
    "plastic": "a plastic model of a {}",
    "plushie": "a plushie {}",
    "rendition": "a rendition of a {}",
    "sculpture": "a sculpture of a {}",
    "sketch": "a sketch of a {}",
    "sticker": "a sticker of a {}",
    "tattoo": "a tattoo of a {}",
    "toy": "a toy {}",
    "video_game": "a video game rendering of a {}",
}


def clip_templates_for_dataset(dataset_name):
    """Return the CLIP prompt ensemble for a supported dataset."""
    if str(dataset_name).lower() == "imagenetr":
        return IMAGENETR_CLIP_TEMPLATES
    return DEFAULT_CLIP_TEMPLATES


def normalize_imagenetr_prefix(path_or_prefix):
    """Map an ImageNet-R filename prefix to a canonical prompt family."""
    value = os.path.splitext(os.path.basename(str(path_or_prefix)))[0].lower()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    value = re.sub(r"_?\d+$", "", value).strip("_")

    # Match the longest aliases first (``plastic_object`` before ``plastic``).
    for alias in sorted(IMAGENETR_PREFIX_ALIASES, key=len, reverse=True):
        if value == alias or value.startswith(alias + "_"):
            return IMAGENETR_PREFIX_ALIASES[alias]
    return None


def _class_subject(class_name, placeholder_token="*"):
    class_name = str(class_name or "").strip().replace("_", " ")
    if class_name:
        return "{} {}".format(class_name, placeholder_token).strip()
    return placeholder_token


def format_class_synthesis_prompt(template, class_name, placeholder_token="*"):
    """Format a class-grounded prompt with a simple English article."""
    clean_name = str(class_name or "").strip().replace("_", " ")
    if not clean_name:
        return str(template).format(
            article="a", class_name="object", placeholder=placeholder_token
        )
    article = "an" if clean_name[0].lower() in "aeiou" else "a"
    return str(template).format(
        article=article, class_name=clean_name, placeholder=placeholder_token
    )


def imagenetr_prefix_prompt(path, class_name, placeholder_token="*"):
    """Build a style-aware prompt for one real ImageNet-R example."""
    prefix = normalize_imagenetr_prefix(path)
    template = IMAGENETR_PREFIX_PROMPT_TEMPLATES.get(prefix)
    if template is None:
        return None
    return template.format(_class_subject(class_name, placeholder_token))


def imagenetr_prompts_for_paths(paths, class_name, fallback="a rendition of *"):
    """Return prompts weighted by the observed filename-prefix histogram.

    Repeating prompts according to the histogram lets ``random.choice`` in the
    sampler reproduce the real class' rough rendition mix without introducing
    a new sampling parameter.
    """
    counts = Counter(normalize_imagenetr_prefix(path) for path in paths)
    counts.pop(None, None)
    if not counts:
        return [str(fallback)]

    subject = _class_subject(class_name)
    prompts = []
    for prefix, count in sorted(counts.items()):
        template = IMAGENETR_PREFIX_PROMPT_TEMPLATES[prefix]
        prompts.extend([template.format(subject)] * int(count))
    return prompts or [str(fallback)]
