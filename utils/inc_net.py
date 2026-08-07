"""Backbone factory and the thin network wrapper shared by all learners."""

from torch import nn


def get_backbone(args, pretrained=False):
    """Build the visual-prompt ViT backbone described by ``backbone_type``."""
    name = args["backbone_type"].lower()

    if name in ("pretrained_vit_b16_224", "vit_base_patch16_224"):
        import timm
        model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        model.out_dim = 768
        return model.eval()

    if "_vpt" in name:
        if args["model_name"] != "asp":
            raise NotImplementedError("VPT backbones are only wired up for model_name='asp'.")
        if name != "pretrained_vit_b16_224_vpt":
            raise NotImplementedError("Unknown VPT backbone {}".format(name))

        from backbone.asp_backbone import build_promptmodel

        vpt_type = "Shallow" if str(args["vpt_type"]).lower() == "shallow" else "Deep"
        model = build_promptmodel(
            modelname="vit_base_patch16_224",
            Prompt_Token_num=args["prompt_token_num"],
            VPT_type=vpt_type,
            args=args,
        )
        model.load_prompt(model.obtain_prompt())
        model.out_dim = 768
        return model.eval()

    raise NotImplementedError("Unknown backbone_type {}".format(name))


class BaseNet(nn.Module):
    """Holds a frozen prompt-tuned backbone plus a replaceable classifier head."""

    def __init__(self, args, pretrained):
        super().__init__()
        self.backbone = get_backbone(args, pretrained)
        self.fc = None
        self._device = args["device"][0]

    @property
    def feature_dim(self):
        return self.backbone.out_dim
