# txt2pic
Wrappers of Text-to-Image generation models.

## Setup

**TL;DR**:
1. Install [PyTorch](https://pytorch.org/get-started/locally/)
2. `pip/conda install gradio`
3. `pip install --upgrade diffusers` or `conda install -c conda-forge diffusers`
4. Got [here](https://huggingface.co/CompVis/stable-diffusion-v1-4) and accept the licence.

More information can be found [here](https://github.com/huggingface/diffusers).

## Stable Diffusion

Run `python stable_diffusion.py` to generate images with [Stable Diffusion](https://stability.ai/blog/stable-diffusion-public-release). This starts an easy to use web interface wrapping most of the [diffusers](https://huggingface.co/CompVis/stable-diffusion-v1-4) library. It requires around 6-8GB of VRAM, depending on the generated image size and other settings.