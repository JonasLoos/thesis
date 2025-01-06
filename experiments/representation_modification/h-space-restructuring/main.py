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
        plt,
        torch,
    )


@app.cell
def __(io, mo):
    def display_image(img, title, width, height):
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
    np,
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

            # setup vae
            if vae == 'auto' and 'sdxl' in self.model_name:
                vae = 'stabilityai/sdxl-vae'
            if isinstance(vae, str) and vae != 'auto':
                vae = AutoencoderKL.from_pretrained(vae).to(self.device)
            self.vae = self.pipeline.vae if vae == 'auto' else vae

            # check h-space dim
            # TODO

        def __call__(self, prompt, steps, guidance_scale, seed: Optional[int] = None, modification: Optional[Callable[[...], torch.Tensor]] = None):
            seed = seed if seed != None else torch.randint(0, 2**32, (1,)).item()
            generator = torch.Generator(device=self.device).manual_seed(seed)
            reprs = []
            imgs = []
            def get_repr(module, input, output):
                reprs.append(output[0].cpu().numpy())
                if modification:
                    return modification(module, input, output)
            def latents_callback(i, t, latents):
                latents = 1 / self.vae.config.scaling_factor * latents.to(dtype=self.vae.dtype)
                image = self.vae.decode(latents).sample[0]
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(1, 2, 0).numpy()
                imgs.extend(self.pipeline.numpy_to_pil(image))
            with self.unet.mid_block.register_forward_hook(get_repr):
                result = self.pipeline(prompt, num_inference_steps = steps, guidance_scale=guidance_scale, callback=latents_callback, callback_steps=1, generator=torch.Generator("cuda").manual_seed(seed))

            return SimpleNamespace(
                seed=seed,
                reprs=np.array(reprs),
                imgs=imgs,
                result=result,
            )
    return SD,


@app.cell
def __(SD):
    sd = SD()
    return sd,


@app.cell
def __(sd):
    x = sd('a cat', 50, 7.5)
    return x,


@app.cell
def __(sdargs, torch):
    total_changes = []
    main_losses = []
    reg_losses = []
    def restructure(sd, masks, change_steps, change_rate, change_until_step, regularization_factor, sd_args):
        @torch.enable_grad
        def mod_fist_step(module, input, output, current_step=[0]):
            # make sure this only executed in the first step
            if current_step[0] > change_until_step: return
            current_step[0] += 1

            # modify output
            hspace = output
            original_hspace = hspace.clone().detach().requires_grad_(False)
            main_losses_ = []
            reg_losses_ = []
            for i in range(change_steps):
                hspace.requires_grad_(True)
                main_loss = 0
                for mask in masks:
                    mask = mask.to(hspace.device)
                    data = hspace.permute(0, 2, 3, 1).reshape(1+1, -1, hspace.shape[1])[mask.reshape(1,-1,1).expand(1+1,-1,1280)].reshape(1+1, -1, 1280)
                    main_loss -= torch.bmm(data, data.transpose(1, 2)).mean()
                    # TODO: try other loss functions, e.g. cosine similarity
                    # main_loss -= F.cosine_similarity(..., ..., dim=1).mean()
                regulatization_loss = ((hspace - original_hspace)**2).mean()
                print(f'[{i}] main loss: {main_loss.item()}, reg. loss: {regulatization_loss.item()}')
                loss = main_loss + regularization_factor * regulatization_loss
                grads = torch.autograd.grad(loss, hspace)[0]
                with torch.no_grad():
                    hspace += change_rate * grads
                main_losses_.append(main_loss.item())
                reg_losses_.append(regulatization_loss.item())

            total_changes.append(((hspace - original_hspace)**2).mean().item())
            main_losses.append(main_losses_)
            reg_losses.append(reg_losses_)

            # return modified hspace
            return hspace

        return sd(**sdargs, modification=mod_fist_step)
    return main_losses, reg_losses, restructure, total_changes


@app.cell
def __(mo):
    prompt_input = mo.ui.text('a cat', label='Prompt')
    change_until_step_slider = mo.ui.slider(0, 50, 1, 10, label='Change Until Step', debounce = True)
    steps_slider = mo.ui.slider(1, 20, 1, 10, label='Change Steps', debounce = True)
    change_rate_slider = mo.ui.slider(-5, 3, 0.1, -1, label='Change Rate', debounce = True)
    reg_slider = mo.ui.slider(-5, 10, 0.1, -1, label='Regularization Factor', debounce = True)
    mo.vstack([prompt_input, change_until_step_slider, steps_slider, change_rate_slider, reg_slider])
    return (
        change_rate_slider,
        change_until_step_slider,
        prompt_input,
        reg_slider,
        steps_slider,
    )


@app.cell
def __(change_rate_slider, mo, prompt_input, reg_slider, steps_slider):
    mo.md(f'Using prompt "{prompt_input.value}" for {steps_slider.value} steps with change rate {10**change_rate_slider.value:.2e} and regularization factor {10**reg_slider.value:.2e}.')
    return


@app.cell
def __(prompt_input, sd):
    sdargs = {
        'prompt': prompt_input.value,
        'steps': 50,
        'guidance_scale': 7.5,
        'seed': 0,
    }
    base = sd(**sdargs)
    return base, sdargs


@app.cell
def __(mo):
    mask_str = mo.ui.code_editor('\n'.join([' '.join(['0']*8)]*8), label = 'Mask')
    mask_str
    return mask_str,


@app.cell
def __(mask_str, torch):
    mask = torch.tensor([[int(x.strip()) for x in line.strip().split(' ')] for line in mask_str.value.split('\n')], dtype=torch.bool)
    return mask,


@app.cell
def __(
    change_rate_slider,
    change_until_step_slider,
    main_losses,
    mask,
    mo,
    reg_losses,
    reg_slider,
    restructure,
    sd,
    sdargs,
    steps_slider,
    total_changes,
):
    total_changes.clear()
    main_losses.clear()
    reg_losses.clear()
    modified = restructure(sd, [mask], steps_slider.value, 10**change_rate_slider.value, change_until_step_slider.value, 10**reg_slider.value, sdargs)
    main_losses
    reg_losses
    mo.md(f'Total changes: {total_changes}')
    return modified,


@app.cell
def __(Image, base, display_image, mask, mo, modified):
    baseimg = display_image(base.result.images[0], 'Original', 256, 256)
    maskimg = display_image(Image.fromarray((255*mask.numpy()).astype('uint8').repeat(256//8, axis=0).repeat(256//8, axis=1)), 'Mask', 256, 256)
    modimg = display_image(modified.result.images[0], 'Modified', 256, 256)
    mo.hstack([baseimg, maskimg, modimg], justify='center')
    return baseimg, maskimg, modimg


@app.cell
def __(main_losses, plt):
    # loss plots
    plt.plot(range(len(main_losses[0])), main_losses[0])
    return


@app.cell
def __(plt, reg_losses):
    plt.plot(range(len(reg_losses[0])), reg_losses[0])
    return


if __name__ == "__main__":
    app.run()
