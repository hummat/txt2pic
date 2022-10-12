from typing import List, Union
import requests
import torch
from PIL import Image
from io import BytesIO
import gradio as gr
from torch import autocast
from diffusers import (DiffusionPipeline, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline,
                       StableDiffusionInpaintPipeline,
                       LMSDiscreteScheduler, DDIMScheduler, PNDMScheduler)

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

"""
url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")
init_image = init_image.resize((768, 512))

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

init_image = Image.open(BytesIO(requests.get(img_url).content)).convert("RGB").resize((512, 512))
mask_image = Image.open(BytesIO(requests.get(mask_url).content)).convert("RGB").resize((512, 512))

scheduler = LMSDiscreteScheduler(beta_start=0.00085,
                                 beta_end=0.012,
                                 beta_schedule="scaled_linear")
scheduler = DDIMScheduler(beta_start=0.00085,
                          beta_end=0.012,
                          beta_schedule="scaled_linear",
                          clip_sample=False,
                          set_alpha_to_one=False)
scheduler = PNDMScheduler(beta_start=0.00085,
                          beta_end=0.012,
                          beta_schedule="scaled_linear")
"""

pipe = StableDiffusionPipeline.from_pretrained(model_id,
                                               # scheduler=scheduler,
                                               torch_dtype=torch.float16,
                                               revision="fp16")
pipe = pipe.to(device)

# pipe.enable_attention_slicing()


def run(prompt: Union[str, List[str]],
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: int = 0) -> Image:
    with autocast(device):
        return pipe(prompt,
                    num_inference_steps=num_inference_steps,
                    # init_image=init_image,
                    # mask_image=mask_image,
                    # strength=0.75,
                    generator=torch.Generator(device=device).manual_seed(int(seed)),
                    guidance_scale=guidance_scale).images[0]


def main():
    interface = gr.Interface(title="Stable Diffusion",
                             fn=run,
                             inputs=[gr.Text(), gr.Slider(0, 500, 50), gr.Slider(0, 10, 7.5), gr.Number(0)],
                             outputs=gr.Image(shape=(512, 512)))
    interface.launch()


if __name__ == "__main__":
    main()
