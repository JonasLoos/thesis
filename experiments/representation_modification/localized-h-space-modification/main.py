import marimo

__generated_with = "0.2.4"
app = marimo.App(width="full")


@app.cell
def __():
    import marimo as mo
    from diffusers import AutoPipelineForText2Image, AutoencoderKL
    import torch
    from sklearn.cluster import DBSCAN
    import numpy as np
    from functools import cache
    import io
    from PIL import Image
    return (
        AutoPipelineForText2Image,
        AutoencoderKL,
        DBSCAN,
        Image,
        cache,
        io,
        mo,
        np,
        torch,
    )


@app.cell
def __(io, mo):
    def mrange(*args, **kwargs):
        return mo.status.progress_bar(range(*args), **kwargs)

    def display_image(img, title, width, height):
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        return mo.vstack([
            mo.md(title),
            mo.image(img_byte_arr.getvalue(), width=width, height=height),
        ])
    return display_image, mrange


@app.cell
def __(mo):
    mo.md("# Localized h-space modification")
    return


@app.cell
def __(AutoPipelineForText2Image, mo, torch):
    with mo.status.spinner(subtitle="Loading model"):
        pipe = AutoPipelineForText2Image.from_pretrained('stabilityai/sdxl-turbo', torch_dtype=torch.float16, variant="fp16").to('cuda')
        pipe.set_progress_bar_config(disable=True)
    return pipe,


@app.cell
def __(cache, pipe, torch):
    @cache
    def get_repr(prompt, seed):
        reprs = []
        def get_repr(module, input, output):
            reprs.append(output[0])
        with pipe.unet.mid_block.register_forward_hook(get_repr):
            img = pipe(prompt, num_inference_steps = 1, guidance_scale=0.0, generator=torch.Generator("cuda").manual_seed(seed)).images[0]
        return reprs[0], img
    return get_repr,


@app.cell
def __(mo):
    base_prompt = mo.ui.text(value='a painting of a cat on a beach', label="Base prompt", full_width=True)
    mod_prompt = mo.ui.text(value='a painting of a dog on a beach', label="Modified prompt", full_width=True)
    num_generations_slider = mo.ui.slider(1, 50, 1, 10, label="Number of generations", debounce=True)
    mo.vstack([base_prompt, mod_prompt, num_generations_slider])
    return base_prompt, mod_prompt, num_generations_slider


@app.cell
def __(num_generations_slider):
    n = num_generations_slider.value
    return n,


@app.cell
def __(base_prompt, get_repr, mod_prompt, mrange, n, torch):
    def get_reprs():
        base_repr_list = []
        base_imgs = []
        for i in mrange(n, title="Generating base representations"):
            repr, img = get_repr(base_prompt.value, i)
            base_repr_list.append(repr.reshape(-1,16*16).T)
            base_imgs.append(img)
        mod_repr_list = []
        mod_imgs = []
        for i in mrange(n, title="Generating modified representations"):
            repr, img = get_repr(mod_prompt.value, i)
            mod_repr_list.append(repr.reshape(-1,16*16).T)
            mod_imgs.append(img)

        base_reprs = torch.stack(base_repr_list)
        mod_reprs  = torch.stack(mod_repr_list)

        return base_reprs, mod_reprs, base_imgs, mod_imgs

    base_reprs, mod_reprs, base_imgs, mod_imgs = get_reprs()
    return base_imgs, base_reprs, get_reprs, mod_imgs, mod_reprs


@app.cell
def __(base_imgs, display_image, mo, mod_imgs):
    mo.vstack([
        mo.md('Example base images'),
        mo.hstack([
            display_image(img, '', 128, 128) for img in base_imgs[:15]
        ]),
        mo.md('Example target images'),
        mo.hstack([
            display_image(img, '', 128, 128) for img in mod_imgs[:15]
        ])
    ])
    return


@app.cell
def __(torch):
    def cossim(a, b):
        # return torch.einsum('jki,lmi->jklm',a,b) / (torch.norm(a, dim=-1)[...,None,None] * torch.norm(b, dim=-1)[None,None,...])
        a = a / torch.norm(a, dim=-1, keepdim=True)
        b = b / torch.norm(b, dim=-1, keepdim=True)
        return torch.einsum('jki,lmi->jklm',a,b)
    return cossim,


@app.cell
def __(base_reprs, cossim, mod_reprs):
    # calculate similarities
    base_similarities = cossim(base_reprs, base_reprs)
    base2mod_similarities = cossim(base_reprs, mod_reprs)
    mod_similarities = cossim(mod_reprs, mod_reprs)
    return base2mod_similarities, base_similarities, mod_similarities


@app.cell
def __():
    # torch.isnan(base_similarities).nonzero()
    # somehow we have nans at some [?,0,?,0]
    return


@app.cell
def __(base_similarities, mo, torch):
    mo.md(
        f"""
        # Find clusters

        Info: Mean max similarity {torch.nan_to_num(base_similarities, nan=-1).max(-1).values.mean():.2f}
        Info: Reasonable values seem to be: 3, 10
        """
    )
    return


@app.cell
def __(mo):
    eps_slider = mo.ui.slider(0.1, 5.0, 0.1, 0.5, label="Epsilon")
    min_samples_slider = mo.ui.slider(1, 10, 1, 5, label="Minimum samples")
    mo.vstack([eps_slider, min_samples_slider])
    return eps_slider, min_samples_slider


@app.cell
def __(
    DBSCAN,
    base_similarities,
    eps_slider,
    min_samples_slider,
    mo,
    n,
    np,
    torch,
):
    # search clusters in the base representations
    _dbscan = DBSCAN(eps=eps_slider.value, min_samples=min_samples_slider.value)
    with mo.status.spinner(subtitle="Clustering"):
        _distances = torch.nan_to_num(base_similarities).reshape(n*16*16,n*16*16).cpu().numpy()
        _distances /= _distances.max()  # against numerical instabilities (?) causing slightly above 1 values
        _distances = (.5 - .5 * _distances)**.5
        clusters = _dbscan.fit_predict(_distances)
    cluster_ids, cluster_counts = np.unique(clusters, return_counts=True)
    return cluster_counts, cluster_ids, clusters


@app.cell
def __(
    base_reprs,
    cluster_counts,
    cluster_ids,
    clusters,
    cossim,
    mo,
    n,
    torch,
):
    _out = f"Total number of clusters: {len(cluster_ids)-1}, number of unclustered repr: {cluster_counts[0]}\n\n"
    _out += 'Cluster | Count | Similarity\n'
    _out += '-|-|-\n'
    for i in cluster_ids[1:]:
        # cosine sim inside cluster
        tmp = base_reprs[torch.tensor(clusters.reshape(n,16*16,1)).expand(n,16*16,1280)==i].reshape(1,-1,1280)
        sim = cossim(tmp, tmp).nanmean()
        assert tmp.shape[1] == cluster_counts[i+1]
        _out += f"{i} |{tmp.shape[1]} | {sim:.2f}\n"
    mo.md(_out)
    return i, sim, tmp


@app.cell
def __(base_reprs, cluster_ids, clusters, cossim, mod_reprs, n, torch):
    # calculate cluster centers and directions
    cluster_centers = torch.stack([base_reprs[torch.tensor(clusters.reshape(n,16*16,1)).expand(n,16*16,1280)==i].reshape(-1,1280).mean(0) for i in cluster_ids[1:]])
    cluster_centers_mod = torch.stack([mod_reprs[torch.arange(n),x,:].mean(0) for x in cossim(cluster_centers.reshape(-1,1,1280), mod_reprs).max(dim=-1).indices.squeeze(1)])
    cluster_directions = cluster_centers_mod - cluster_centers
    return cluster_centers, cluster_centers_mod, cluster_directions


@app.cell
def __(mo):
    mo.md(f"# Directions")
    return


@app.cell
def __(base_reprs, cossim, mod_reprs, torch):
    def get_best_dir_index(factor, index):
        base_similarity_scores = cossim(base_reprs, base_reprs).max(dim=-1).values.mean(dim=-1)
        base2mod_similarity_scores = cossim(base_reprs, mod_reprs).max(dim=-1).values.mean(dim=-1)
        score = base_similarity_scores - factor * base2mod_similarity_scores
        score = torch.nan_to_num(score, nan=-1)
        tmp = score.argmax()
        return int(tmp // 16**2), int(tmp % 16**2)
    return get_best_dir_index,


@app.cell
def __(mo):
    new_prompt_text = mo.ui.text(value='a painting of a cat on a beach', label="Prompt", full_width=True)
    use_clusters_checkbox = mo.ui.checkbox(value=True, label="Use clusters")
    mo.vstack([new_prompt_text, use_clusters_checkbox])
    return new_prompt_text, use_clusters_checkbox


@app.cell
def __(mo):
    pasted_direction = mo.ui.text(label="Paste custom direction here", full_width=True)
    pasted_direction
    return pasted_direction,


@app.cell
def __(cluster_directions, mo, use_clusters_checkbox):
    score_factor_slider = mo.ui.slider(0.0, 2.0, 0.1, 1.0, label="Score Modification Factor", debounce=True)
    num_indices = cluster_directions.shape[0]-1 if use_clusters_checkbox.value else 10
    index_slider = mo.ui.slider(0, num_indices, 1, 0, label="Direction to use", debounce=True)
    direction_factor_slider = mo.ui.slider(-10, 10, 1, 0, label="Direction Strength Factor", debounce=True)
    mo.vstack([score_factor_slider, index_slider, direction_factor_slider]) if not use_clusters_checkbox.value else mo.vstack([index_slider, direction_factor_slider])
    return (
        direction_factor_slider,
        index_slider,
        num_indices,
        score_factor_slider,
    )


@app.cell
def __(cache, cossim, pipe, torch):
    @cache
    def generate_image(prompt, base, direction, strength, seed=0):
        tmp = []
        def set_repr(module, input, output):
            similarities = cossim(base.reshape(1,1,1280), output.reshape(1,1280,16*16).mT)[0,0,0,:].reshape(1,1,16,16)
            print(f'Number of nans: {torch.isnan(similarities).sum()}')
            similarities = torch.nan_to_num(similarities, nan=0)
            tmp.append(similarities)
            return output + strength * direction.reshape(1,1280,1,1) * (similarities > 0) * similarities**4
        with pipe.unet.mid_block.register_forward_hook(set_repr):
            img = pipe(prompt, num_inference_steps = 1, guidance_scale=0.0, generator=torch.Generator("cuda").manual_seed(seed)).images[0]
        return img, tmp[0]
    return generate_image,


@app.cell
def __(
    base_reprs,
    cluster_centers,
    cluster_directions,
    cossim,
    e,
    get_best_dir_index,
    index_slider,
    mod_reprs,
    n,
    pasted_direction,
    score_factor_slider,
    torch,
    use_clusters_checkbox,
):
    if pasted_direction.value:
        try:
            best_base_repr = torch.tensor([float(x.strip()) for x in pasted_direction.value.split(',')]).to('cuda', dtype=torch.float16)
            assert best_base_repr.shape == (1280,), f"Expected shape (1280,), got {best_base_repr.shape}"
            _base_indices = cossim(best_base_repr.reshape(1,1,-1), base_reprs)[0,0,:,:].argmax(axis=-1)
            _mod_indices = cossim(best_base_repr.reshape(1,1,-1), mod_reprs)[0,0,:,:].argmax(axis=-1)
            best_direction = mod_reprs[torch.arange(n),_mod_indices,:].mean(axis=0) - base_reprs[torch.arange(n),_base_indices,:].mean(axis=0)
            best_dir_index = 'custom'
        except Exception as e:
            best_direction = torch.zeros(1280).to('cuda', dtype=torch.float16)
            best_dir_index = f'error: {e}'
    elif use_clusters_checkbox.value:
        best_dir_index = f'cluster {index_slider.value}'
        best_base_repr = cluster_centers[index_slider.value]
        best_direction = cluster_directions[index_slider.value]
    else:
        best_dir_index = get_best_dir_index(score_factor_slider.value, index_slider.value)
        best_base_repr = base_reprs[best_dir_index]
        best_direction = ...
    return best_base_repr, best_dir_index, best_direction


@app.cell
def __(best_dir_index, mo):
    mo.md(f"Index of best direction found: {best_dir_index}")
    return


@app.cell
def __(
    Image,
    best_base_repr,
    best_direction,
    direction_factor_slider,
    display_image,
    generate_image,
    mo,
    new_prompt_text,
    np,
    pasted_direction,
    use_clusters_checkbox,
):
    strength = direction_factor_slider.value
    if pasted_direction.value:
        strength *= 500
    elif use_clusters_checkbox.value:
        strength *= 10
    img, similarities = generate_image(new_prompt_text.value, best_base_repr, best_direction, strength)
    _sims = (similarities[0,0] > 0)*similarities[0,0]**4
    _sims /= _sims.max()

    mo.hstack([
        display_image(img, 'modified image', 512, 512),
        display_image(Image.fromarray((_sims.cpu().numpy() * 255).astype(np.uint8).repeat(512//16, axis=0).repeat(512//16, axis=1)), 'similarities', 512, 512)
    ], justify='center')
    return img, similarities, strength


if __name__ == "__main__":
    app.run()
