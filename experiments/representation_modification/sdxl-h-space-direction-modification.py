'''
Streamlit app for experimenting with stable diffusion unet representations.
Create a direction in the representation space for a given prompt and modify the generated image accordingly.

Run with: `streamlit run sdxl-h-space-direction-modification.py`

Tested using Python 3.11 and streamlit 1.29.0. Requires a GPU with CUDA support.

author: Jonas Loos
'''

import streamlit as st
from diffusers import AutoPipelineForText2Image
import torch
import numpy as np
from tqdm import trange


# config
model = 'stabilityai/sdxl-turbo'  # hugginface model to use
modification_strength = 20  # 20 seems to work well
attribute_generation_steps = 50
attribute_generation_diffusion_steps = 4


@st.cache_resource
def get_pipeline():
    '''Load and cache the stable diffusion model.'''
    pipe = AutoPipelineForText2Image.from_pretrained(model, torch_dtype=torch.float16, variant="fp16").to('cuda')
    pipe.set_progress_bar_config(disable=True)
    return pipe

@st.cache_resource
def get_seed():
    return torch.randint(0, 100000, (1,)).item()

pipe = get_pipeline()
seed = get_seed()

@st.cache_resource
def get_attr(text, attr):
    '''Get the attribute direction for a given prompt and attribute.'''
    reprs_base = []
    reprs_attr = []
    def get_repr_base(module, input, output): reprs_base.append(output)
    def get_repr_attr(module, input, output): reprs_attr.append(output)
    try:
        for i in trange(attribute_generation_steps, desc='generating attribute direction'):
            with torch.no_grad(), pipe.unet.mid_block.register_forward_hook(get_repr_base):
                pipe(prompt = text, num_inference_steps = attribute_generation_diffusion_steps, guidance_scale=0.0)
            with torch.no_grad(), pipe.unet.mid_block.register_forward_hook(get_repr_attr):
                pipe(prompt = f'{text}, {attr}', num_inference_steps = attribute_generation_diffusion_steps, guidance_scale=0.0)
        diff = torch.stack(reprs_attr).mean(axis=0) - torch.stack(reprs_base).mean(axis=0)
    except:
        diff = torch.zeros((1,1280,16,16), device='cuda', dtype=torch.float16)
    return diff

st.title('SDXL direction modifier')

# image display
img = st.empty()
if 'last_img' in st.session_state:
    img.image(st.session_state.last_img, use_column_width=True)
else:
    img.image(np.zeros((512,512,3)), use_column_width=True)

# input
text = st.text_input('Enter prompt', 'a portrait of a panda scientist')
num_steps = st.slider('Number of inference steps', 1, 6, 2)
if st.button('new seed'):
    get_seed.clear()
    seed = get_seed()

attr_text = st.text_input('Enter attribute to modify', 'doing dangerous experiments')
attr_scaling = st.slider('Attribute scaling', -1., 1., 0.)
attr_vec = get_attr(text, attr_text)

# generate image
def adjust_repr(module, input, output):
    change = (attr_scaling * modification_strength) * attr_vec
    return output + change
with torch.no_grad(), pipe.unet.mid_block.register_forward_hook(adjust_repr):
    last_img = pipe(prompt = text, num_inference_steps = num_steps, guidance_scale = 0.0, generator = torch.Generator("cuda").manual_seed(seed)).images[0]
img.image(last_img, use_column_width=True)
st.session_state.last_img = last_img
