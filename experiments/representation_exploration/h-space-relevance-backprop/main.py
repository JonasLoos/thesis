import marimo

__generated_with = "0.2.4"
app = marimo.App(width="full")


@app.cell
def __():
    import marimo as mo
    from diffusers import AutoPipelineForText2Image, AutoencoderKL
    import torch
    import torch.nn.functional as F
    import numpy as np
    from functools import cache
    import io
    from PIL import Image
    from typing import Optional, Callable
    from types import SimpleNamespace
    from functools import cache
    import matplotlib.pyplot as plt
    from transformers import pipeline
    return (
        AutoPipelineForText2Image,
        AutoencoderKL,
        Callable,
        F,
        Image,
        Optional,
        SimpleNamespace,
        cache,
        io,
        mo,
        np,
        pipeline,
        plt,
        torch,
    )


@app.cell
def __(io, mo):
    def display_image(img, title='', width=512, height=512):
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        return mo.vstack([
            mo.md(title),
            mo.image(img_byte_arr.getvalue(), width=width, height=height),
        ])
    return display_image,


@app.cell
def __(
    AutoPipelineForText2Image,
    AutoencoderKL,
    Callable,
    Optional,
    SimpleNamespace,
    torch,
):
    class SD:
        known_models = {
            'SD1.5': 'runwayml/stable-diffusion-v1-5',
            'SD2.1': 'stabilityai/stable-diffusion-2-1',
            'SD-Turbo': 'stabilityai/sd-turbo',
            'SDXL-Turbo': 'stabilityai/sdxl-turbo',
        }
        def __init__(self, model_name: str = 'SD1.5', device: str = 'auto', vae: str | AutoencoderKL = 'auto'):
            self.model_name = self.known_models.get(model_name, model_name)
            self.device = device if device != 'auto' else 'cuda' if torch.cuda.is_available() else 'cpu'

            # setup pipeline
            self.pipeline = AutoPipelineForText2Image.from_pretrained(self.model_name, torch_dtype=torch.float16).to(self.device)
            self.unet = self.pipeline.unet
            self.call_pipeline = self.pipeline
            # if preserve_grad:
                # TODO: this doesn't seem to work well, due to extreme gpu memory usage. It seems to use roughly 20GB for 15 steps
                # ignore pipeline.__call__ decorators, i.e. torch.no_grad
                # disable require_grad for all parameters
                # for param in self.unet.parameters():
                #     param.requires_grad = False
                # for param in self.pipeline.vae.parameters():
                #     param.requires_grad = False

            # setup vae
            if vae == 'auto' and 'sdxl' in self.model_name:
                vae = 'stabilityai/sdxl-vae'
            if isinstance(vae, str) and vae != 'auto':
                vae = AutoencoderKL.from_pretrained(vae).to(self.device)
            self.vae = self.pipeline.vae if vae == 'auto' else vae
            # TODO: maybe rather updast vae like in https://github.com/huggingface/diffusers/blob/12004bf3a7d3e77eafe3dd8fad1d458d8e001fab/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L1313

            # check h-space dim
            # TODO

        def __call__(self, prompt, steps, guidance_scale, seed: Optional[int] = None, modification: Optional[Callable[[...], torch.Tensor]] = None, preserve_grad: bool = False):
            seed = seed if seed != None else torch.randint(0, 2**32, (1,)).item()
            generator = torch.Generator(device=self.device).manual_seed(seed)
            call_pipeline = self.call_pipeline = lambda *args, **kwargs: self.pipeline.__class__.__call__.__wrapped__(self.pipeline, *args, **kwargs) if preserve_grad else self.pipeline
            reprs = []
            imgs = []
            def get_repr(module, input, output):
                reprs.append(output[0])
                if modification:
                    return modification(module, input, output)
            @torch.no_grad
            def latents_callback(i, t, latents):
                latents = 1 / self.vae.config.scaling_factor * latents.detach().to(dtype=self.vae.dtype)
                image = self.vae.decode(latents).sample[0].detach()
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(1, 2, 0).numpy()
                imgs.extend(self.pipeline.numpy_to_pil(image))
            with self.unet.mid_block.register_forward_hook(get_repr):
                result = call_pipeline(prompt, num_inference_steps = steps, guidance_scale=guidance_scale, callback=latents_callback, callback_steps=1, generator=torch.Generator("cuda").manual_seed(seed), output_type='latent')
            # cast images to same dtype as vae
            result_tensor = self.vae.decode(result.images.to(dtype=self.vae.dtype) / self.vae.config.scaling_factor).sample
            result_image = self.pipeline.image_processor.postprocess(result_tensor.detach(), output_type='pil')

            return SimpleNamespace(
                seed=seed,
                reprs=reprs,
                imgs=imgs,
                result_latent=result.images[0],
                result_tensor=result_tensor[0],
                result_image=result_image[0],
            )
    return SD,


@app.cell
def __(SD):
    sd = SD('SDXL-Turbo')
    return sd,


@app.cell
def __(mo):
    mo.md('## Base Image')
    return


@app.cell
def __(mo):
    regenerate_button = mo.ui.button(label='Regenerate')
    regenerate_button
    return regenerate_button,


@app.cell
def __(regenerate_button, sd):
    regenerate_button
    generation = sd('a cat', 2, 0, preserve_grad=True)
    # generation = sd('a cat', 50, 7.5)
    return generation,


@app.cell
def __(display_image, generation):
    display_image(generation.result_image)
    return


@app.cell
def __(pipeline):
    sam_generator = pipeline("mask-generation", model="facebook/sam-vit-huge", device='cuda')
    return sam_generator,


@app.cell
def __(generation, sam_generator):
    outputs = sam_generator(generation.result_image, points_per_batch=64)
    return outputs,


@app.cell
def __(Image, np):
    def masks_on_image(raw_image, masks, colors=None):
        raw_image = raw_image.convert("RGBA")
        for i, mask in enumerate(masks):
            if colors is None:
                color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            else:
                color = colors[i]
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            mask_image = (mask_image * 255).astype(np.uint8)
            mask_image = Image.fromarray(mask_image, "RGBA")
            raw_image = Image.alpha_composite(raw_image, mask_image)
        return raw_image
    return masks_on_image,


@app.cell
def __(mo):
    mo.md(f'## All Masks')
    return


@app.cell
def __(display_image, generation, masks_on_image, outputs):
    display_image(masks_on_image(generation.result_image, outputs['masks']))
    return


@app.cell
def __(mo):
    mo.md('## Select Mask')
    return


@app.cell
def __(mo, outputs):
    mask_slider = mo.ui.slider(0, len(outputs['masks']) - 1, 1, 0, label='Mask Index')
    mask_slider
    return mask_slider,


@app.cell
def __(
    display_image,
    generation,
    mask_slider,
    masks_on_image,
    np,
    outputs,
):
    display_image(masks_on_image(generation.result_image, [outputs['masks'][mask_slider.value]], colors=np.array([[1, 0, 0, 0.6]])))
    return


@app.cell
def __(mo):
    mo.md('## Get relevant pixels in h-space')
    return


@app.cell
def __(generation, mask_slider, outputs, torch):
    mask = outputs['masks'][mask_slider.value]
    grads = torch.autograd.grad(outputs=generation.result_latent.sum(), inputs=[generation.reprs[0]])
    # grads = torch.autograd.grad(outputs=generation.result_tensor[:,mask].sum(), inputs=[generation.reprs[0]])
    # problem: RuntimeError: One of the differentiated Tensors appears to not have been used in the graph. Set allow_unused=True if this is the desired behavior.
    grads
    return grads, mask


@app.cell
def __(generation):
    # Try to debug where the gradient flow stops
    # doesn't seem to work. Too big graph?

    import sys

    # Increase the recursion limit; the default is usually 1000
    sys.setrecursionlimit(5000)  # Or higher, depending on your needs

    # Your make_dot code here
    from torchviz import make_dot
    make_dot(generation.result_latent, params=dict(result_latent=generation.reprs[1])).render("computation_graph", format="png")
    return make_dot, sys


if __name__ == "__main__":
    app.run()
