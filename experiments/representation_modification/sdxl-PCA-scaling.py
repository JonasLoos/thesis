'''
Streamlit app for experimenting with stable diffusion unet representations.
Adjust the principal components of the representation to see how it affects the generated image.

Run with: `streamlit run sdxl-PCA-scaling.py`

Tested using Python 3.11 and streamlit 1.29.0. Requires a GPU with CUDA support.

author: Jonas Loos
'''

import streamlit as st
from diffusers import AutoPipelineForText2Image
import torch
import numpy as np
from sklearn.decomposition import PCA


# config
num_sliders = 10  # number of modifiable PCs
model = 'stabilityai/sdxl-turbo'  # hugginface model to use
modification_strength = 100  # 100 seems to work well


@st.cache_resource
def get_pipeline():
    '''Load and cache the stable diffusion model.'''
    return AutoPipelineForText2Image.from_pretrained(model, torch_dtype=torch.float16, variant="fp16").to('cuda')

@st.cache_resource
def get_seed():
    return torch.randint(0, 100000, (1,)).item()

pipe = get_pipeline()
seed = get_seed()

def reset_sliders():
    '''Reset all sliders to 0.'''
    for i in range(num_sliders):
        st.session_state[f'pc{i}'] = 0

@st.cache_resource
def generate_pcs(text):
    '''Generate principal components of the representation for the given text.'''
    reprs = []
    def get_repr(module, input, output):
        # take mean over spatial dimensions
        reprs.append(output.mean(dim=(0,2,3)).detach().cpu().numpy())
    for _ in range(20):
        with torch.no_grad(), pipe.unet.mid_block.register_forward_hook(get_repr):
            pipe(prompt = text, num_inference_steps = 2, guidance_scale=0.0)
    pca = PCA(n_components=num_sliders)
    pca.fit(reprs)
    reset_sliders()
    return pca.components_

st.title('SDXL PCA scaling')
col1, col2 = st.columns([2,1])

with col1:
    # image display
    img = st.empty()
    if 'last_img' in st.session_state:
        img.image(st.session_state.last_img, use_column_width=True)
    else:
        img.image(np.zeros((512,512,3)), use_column_width=True)

    # input
    text = st.text_input('Enter prompt')
    num_steps = st.slider('Number of inference steps', 1, 6, 1)
    if st.button('new seed'):
        get_seed.clear()
        generate_pcs.clear()
        seed = get_seed()
        pcs = generate_pcs(text)
    pcs = generate_pcs(text)

with col2:
    # sliders
    'Adjust representation (PCs):'
    st.button('reset', on_click=reset_sliders)
    pc_scalings = np.array([st.slider(f'PC{i}',-1., 1., 0., key=f'pc{i}', label_visibility='collapsed') for i in range(num_sliders)])[:,None]

# generate image
def adjust_repr(module, input, output):
    change = torch.tensor((pcs*pc_scalings).sum(axis=0).reshape(1, -1, 1, 1)).half().cuda() * modification_strength
    return output + change
with torch.no_grad(), pipe.unet.mid_block.register_forward_hook(adjust_repr):
    last_img = pipe(prompt = text, num_inference_steps = num_steps, guidance_scale = 0.0, generator = torch.Generator("cuda").manual_seed(seed)).images[0]
img.image(last_img, use_column_width=True)
st.session_state.last_img = last_img
