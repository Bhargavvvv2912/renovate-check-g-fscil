# Implementation references

This repository implements the method described in **Learning to Curate What
You Generate: Generalizable Few-Shot Class-Incremental Learning**. The method is
built upon and improves the ASP framework, while referring to DDDR for
synthetic generation and CLIP-powered data selection for sample curation.

The implementation also adapts ideas and code structure from the following
research repositories:

- [FSCIL-ASP](https://github.com/DawnLIU35/FSCIL-ASP), the foundation of this
  implementation and the official code of
  *Few-Shot Class Incremental Learning with Attention-Aware Self-Adaptive
  Prompt*: prompt-tuned ViT backbone, task-invariant/task-specific prompts,
  prompt-memory update, prototype classifier, information-bottleneck loss, and
  anchor regularization.
- [DDDR](https://github.com/jinglin-liang/DDDR), the implementation of
  *Diffusion-Driven Data Replay: A Novel Approach to Combat Forgetting in
  Federated Class Continual Learning*: frozen latent-diffusion
  initialization, class inversion, learned class-condition embeddings, DDIM
  synthesis, and task/class-organized synthetic replay caches. The federated
  training components of DDDR are not used here.
- [A CLIP-Powered Framework for Robust and Generalizable Data Selection](https://github.com/Jackbrocp/clip-powered-data-selection):
  CLIP image/text semantic scoring, class-wise nearest-neighbor diversity, and
  learnable ratio-constrained selection masks, referenced here for synthetic
  candidate curation.
- [latent-diffusion](https://github.com/CompVis/latent-diffusion) and
  [OpenAI CLIP](https://github.com/openai/CLIP): underlying generative and
  vision-language model implementations.

Please cite this work and the relevant referenced papers. Upstream code,
datasets, models, and checkpoints remain subject to their respective terms.
