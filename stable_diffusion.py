import random
from typing import List, Tuple, Union, Dict

import diffusers
import gradio as gr
import torch
from PIL import Image


def get_scheduler(scheduler_name: str):
    scheduler = getattr(diffusers, scheduler_name)
    if scheduler_name == "DDIMScheduler":
        return scheduler(beta_start=0.00085,
                         beta_end=0.012,
                         beta_schedule="scaled_linear",
                         clip_sample=False,
                         set_alpha_to_one=False)
    return scheduler(beta_start=0.00085,
                     beta_end=0.012,
                     beta_schedule="scaled_linear")


def image_grid(images, rows, cols):
    assert len(images) == rows * cols

    w, h = images[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


pipe = None
current_model_id = None
current_revision = None
current_scheduler = None


def run(prompt: str,
        init_image: Union[Image.Image, Dict[str, Image.Image]] = None,
        strength: float = 0.8,
        num_images: int = 1,
        sequential: bool = True,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        enable_attention_slicing: bool = False,
        pipe_name: str = "StableDiffusionPipeline",
        model_id: str = "CompVis/stable-diffusion-v1-4",
        scheduler_name: str = "PNDMScheduler",
        revision: str = "fp16",
        device: str = "cuda",
        seed: int = 123456789) -> Tuple[Image.Image, List[Image.Image]]:
    print(f"Running {pipe_name} with model {model_id} and revision {revision} on device {device}...")

    if pipe_name != "StableDiffusionPipeline" and init_image is None:
        raise gr.Error("Please provide an initial image.")

    rows = int(num_images ** 0.5)
    cols = int(num_images / rows)
    if num_images != rows * cols:
        raise gr.Error(f"Invalid number of images: {num_images}.")

    if device == "cpu" and revision == "fp16":
        raise gr.Error(f"Cannot use revision {revision} with device {device}.")

    global pipe, current_model_id, current_revision, current_scheduler

    new_pipe = pipe is None or type(pipe).__name__ != pipe_name
    if new_pipe or current_model_id != model_id or current_revision != revision or current_scheduler != scheduler_name:
        current_model_id = model_id
        current_revision = revision
        current_scheduler = scheduler_name

        pipe = getattr(diffusers, pipe_name)
        pipe = pipe.from_pretrained(model_id,
                                    scheduler=get_scheduler(scheduler_name),
                                    torch_dtype=torch.float16 if revision == "fp16" else torch.float32,
                                    revision=revision)

    if enable_attention_slicing:
        pipe.enable_attention_slicing()
    else:
        pipe.disable_attention_slicing()

    pipe = pipe.to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    prompts = [prompt] * num_images

    mask_image = None
    if init_image is not None:
        if pipe_name == "StableDiffusionInpaintPipeline":
            mask_image = init_image["mask"]
            init_image = init_image["image"]
        if height != -1 and width != -1:
            init_image = init_image.resize((width, height))
            if mask_image is not None:
                mask_image = mask_image.resize((width, height))

    with torch.autocast(device, enabled=device == "cuda" and revision == "fp16"):
        if sequential:
            images = list()
            for prompt in prompts:
                if pipe_name == "StableDiffusionPipeline":
                    images.append(pipe(prompt,
                                       height=height,
                                       width=width,
                                       num_inference_steps=num_inference_steps,
                                       guidance_scale=guidance_scale,
                                       generator=generator).images[0])
                elif pipe_name == "StableDiffusionImg2ImgPipeline":
                    images.append(pipe(prompt,
                                       init_image=init_image,
                                       strength=strength,
                                       num_inference_steps=num_inference_steps,
                                       guidance_scale=guidance_scale,
                                       generator=generator).images[0])
                elif pipe_name == "StableDiffusionInpaintPipeline":
                    images.append(pipe(prompt,
                                       init_image=init_image,
                                       mask_image=mask_image,
                                       strength=strength,
                                       num_inference_steps=num_inference_steps,
                                       guidance_scale=guidance_scale,
                                       generator=generator).images[0])
        else:
            if pipe_name == "StableDiffusionPipeline":
                images = pipe(prompts,
                              height=height,
                              width=width,
                              num_inference_steps=num_inference_steps,
                              guidance_scale=guidance_scale,
                              generator=generator).images
            elif pipe_name == "StableDiffusionImg2ImgPipeline":
                images = pipe(prompts,
                              init_image=init_image,
                              strength=strength,
                              num_inference_steps=num_inference_steps,
                              guidance_scale=guidance_scale,
                              generator=generator).images
            elif pipe_name == "StableDiffusionInpaintPipeline":
                images = pipe(prompts,
                              init_image=init_image,
                              mask_image=mask_image,
                              strength=strength,
                              num_inference_steps=num_inference_steps,
                              guidance_scale=guidance_scale,
                              generator=generator).images

    if len(images) == 1:
        return images[0], images
    return image_grid(images, rows, cols), images


def main():
    prompt = gr.Textbox(label="Prompt")
    init_image = gr.Image(shape=(512, 512), tool="sketch", type="pil", label="Initial Image", visible=False)
    strength = gr.Slider(0, 1, 0.8, step=0.01, label="Strength (amount of transformation)", visible=False)
    num_images = gr.Slider(1, 9, 1, step=1, label="Number of images")
    sequential = gr.Checkbox(True, label="Generate sequentially")
    height = gr.Number(512, precision=0, label="Height")
    width = gr.Number(512, precision=0, label="Width")
    num_inference_steps = gr.Slider(0, 500, 50, step=1, label="Number of inference steps")
    guidance_scale = gr.Slider(0, 10, 7.5, step=0.1, label="Guidance scale")
    enable_attention_slicing = gr.Checkbox(False, label="Enable attention slicing")
    pipe_name = gr.Radio(["StableDiffusionPipeline",
                          "StableDiffusionImg2ImgPipeline",
                          "StableDiffusionInpaintPipeline"],
                         value="StableDiffusionPipeline",
                         label="Pipeline",
                         visible=False)
    model_id = gr.Dropdown(["CompVis/stable-diffusion-v1-1",
                            "CompVis/stable-diffusion-v1-2",
                            "CompVis/stable-diffusion-v1-3",
                            "CompVis/stable-diffusion-v1-4"],
                           value="CompVis/stable-diffusion-v1-4",
                           label="Model ID")
    scheduler_name = gr.Dropdown(["PNDMScheduler",
                                  "LMSDiscreteScheduler",
                                  "DDIMScheduler"],
                                 value="PNDMScheduler",
                                 label="Scheduler")
    revision = gr.Radio(["fp16", "main"], value="fp16", label="Revision")
    device = gr.Radio(["cuda", "cpu"], value="cuda", label="Device")
    seed = gr.Number(random.randint(0, 10 ** 9), precision=0, label="Seed")

    inputs = [prompt,
              init_image,
              strength,
              num_images,
              sequential,
              height,
              width,
              num_inference_steps,
              guidance_scale,
              enable_attention_slicing,
              pipe_name,
              model_id,
              scheduler_name,
              revision,
              device,
              seed]
    outputs = [gr.Image(label="Generated Image(s)"), gr.Gallery(label="Image Gallery")]

    with gr.Blocks() as interface:
        gr.Markdown("# Stable Diffusion")

        tabs = list()
        for tab_label in ["Text to Image", "Image to Image", "Image Inpainting"]:
            for i in inputs:
                i.unrender()
            for o in outputs:
                o.unrender()

            with gr.Tab(tab_label) as tab:
                tabs.append(tab)
                with gr.Row():
                    with gr.Column():
                        prompt.render()
                        init_image.render()
                        strength.render()
                        with gr.Row():
                            num_images.render()
                            sequential.render()
                        with gr.Row():
                            height.render()
                            width.render()
                        with gr.Accordion("Advanced", open=False):
                            advanced = [num_inference_steps,
                                        guidance_scale,
                                        enable_attention_slicing,
                                        pipe_name,
                                        model_id,
                                        scheduler_name,
                                        revision,
                                        device,
                                        seed]
                            for item in advanced:
                                item.render()
                    with gr.Column():
                        for output in outputs:
                            output.render()

        def select_txt2img_pipe():
            return [gr.update(value="StableDiffusionPipeline"),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(value=512),
                    gr.update(value=512)]

        def select_img2img_pipe():
            return [gr.update(value="StableDiffusionImg2ImgPipeline"),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(value=-1),
                    gr.update(value=-1)]

        def select_inpaint_pipe():
            return [gr.update(value="StableDiffusionInpaintPipeline"),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(value=-1),
                    gr.update(value=-1)]

        tabs[0].select(fn=select_txt2img_pipe, inputs=[], outputs=[pipe_name, init_image, strength, height, width])
        tabs[1].select(fn=select_img2img_pipe, inputs=[], outputs=[pipe_name, init_image, strength, height, width])
        tabs[2].select(fn=select_inpaint_pipe, inputs=[], outputs=[pipe_name, init_image, strength, height, width])

        inference_btn = gr.Button("Generate Image(s)")
        inference_btn.click(fn=run,
                            inputs=inputs,
                            outputs=outputs)

    interface.launch()


if __name__ == "__main__":
    main()
