import gradio as gr
from sdhelper import SD
from functools import cache
import torch
from PIL import Image

# global variables
if gr.NO_RELOAD: sd = SD()
reprs: list[torch.Tensor | None] = [None, None]
last_clicked = [0,0,0]
repr_shape = [[1,1,1], [1,1,1]]


def calc_similarities(img1, img2, scale_to_default):
    '''Calculate the cosine similarities between the last clicked tile and the representations of the images.'''
    img_id, i, j = last_clicked
    n, m = repr_shape[img_id][-2:]
    if reprs[img_id] is None: return None, None
    i = max(0, min(i, n-1))
    j = max(0, min(j, m-1))
    tile = reprs[img_id][:, i, j][:, None, None]

    # update similarity image 1
    results = []
    for repr, img in zip(reprs, [img1, img2]):
        if repr is not None:
            # calculate similarity
            similarity = (repr * tile).sum(0) / (repr.norm(2, dim=0) * tile.norm(2) + 1e-6)
            # add color channel
            similarity = torch.stack([similarity]*3, 2)
            # mark the clicked tile
            if repr is reprs[img_id]: similarity[i,j,1:] = 0
            # clamp to (-1, 1), convert and prevent anti-aliasing
            y, x = (512, 512) if scale_to_default else img.shape[:2]
            results.append(similarity.float().clamp(-1, 1).cpu().numpy().repeat(y//n, axis=0).repeat(x//m, axis=1))
        else:
            results.append(None)

    return results


with gr.Blocks() as demo:
    ZERO = gr.Number(value=0, visible=False)
    ONE = gr.Number(value=1, visible=False)

    gr.Markdown("## SD Similarity Explorer")

    position = gr.Dropdown(label="Position for representation extraction", choices=sd.available_extract_positions, value=sd.available_extract_positions[0])
    step = gr.Slider(label="Step for representation extraction (higher -> more noise)", minimum=0, maximum=999, value=100)
    scale_to_default = gr.Checkbox(label="Crop and scale image to default shape of (512,512)", value=False)

    with gr.Row():
        img1 = gr.Image(label="Upload First Image")
        img2 = gr.Image(label="Upload Second Image")

    with gr.Row():
        output_img1 = gr.Image(label="Processed Image 1")
        output_img2 = gr.Image(label="Processed Image 2")


    def update_repr(img1, img2, position, step, scale_to_default, img_id, calc_sim=True):
        img = img1 if img_id == 0 else img2
        if img is not None:
            img = Image.fromarray(img)
            # scale the image to default shape if needed
            if scale_to_default:
                tmp = min(img.size)
                img = img.crop((img.size[1]//2-tmp//2, img.size[0]//2-tmp//2, img.size[1]//2+tmp//2, img.size[0]//2+tmp//2)).resize((512, 512))
            # calculate the representations
            reprs[img_id] = sd.img2repr(img, [position], step=int(step))[position][0]

            # update the representation shape and scale the last clicked tile to preserve the position
            new_shape = reprs[img_id].shape
            last_clicked[1] = int(last_clicked[1] * (new_shape[-2] / repr_shape[img_id][-2]))
            last_clicked[2] = int(last_clicked[2] * (new_shape[-1] / repr_shape[img_id][-1]))
            repr_shape[img_id] = new_shape

        else:
            reprs[img_id] = None

        if calc_sim:
            return calc_similarities(img1, img2, scale_to_default)


    def update_repr_both(img1, img2, position, step, scale_to_default):
        update_repr(img1, img2, position, step, scale_to_default, 0, False)
        update_repr(img1, img2, position, step, scale_to_default, 1, False)
        return calc_similarities(img1, img2, scale_to_default)


    def on_select(img1, img2, scale_to_default, img_id, evt : gr.SelectData):
        img = img1 if img_id == 0 else img2
        n, m = repr_shape[img_id][-2:]
        if scale_to_default:
            tmp = min(img.shape[:2])
            i = (evt.index[1] - (img.shape[0] - tmp) // 2) * n // (img.shape[0] - (img.shape[0] - tmp))
            j = (evt.index[0] - (img.shape[1] - tmp) // 2) * m // (img.shape[1] - (img.shape[1] - tmp))
            i = max(0, min(i, n-1))
            j = max(0, min(j, m-1))
        else:
            i = evt.index[1] * n // img.shape[0]
            j = evt.index[0] * m // img.shape[1]
        print(f'Clicked: ({evt.index[1]},{evt.index[0]}) / ({img.shape[0]}, {img.shape[1]}) -> ({i},{j}) / ({n},{m})')
        last_clicked[:] = [img_id, i, j]
        return calc_similarities(img1, img2, scale_to_default)


    # event listeners
    inp = [img1, img2, position, step, scale_to_default]
    out = [output_img1, output_img2]
    img1.change(update_repr, inp+[ZERO], out)
    img2.change(update_repr, inp+[ONE], out)
    position.change(update_repr_both, inp, out)
    step.change(update_repr_both, inp, out)
    scale_to_default.change(update_repr_both, inp, out)
    img1.select(on_select, [img1, img2, scale_to_default, ZERO], out)
    img2.select(on_select, [img1, img2, scale_to_default, ONE], out)


demo.launch()
