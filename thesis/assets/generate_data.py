import sys
from sdhelper import SD
import torch
import pickle
from tqdm import tqdm, trange
from pathlib import Path
from time import sleep
from datasets import load_dataset
from PIL import Image, ImageFilter
import numpy as np
from typing import Callable


############################################################################
# data generation functions
############################################################################


def generate_norms(path: Path):
    models = ['SD-1.5', 'SD-2.1', 'SD-Turbo', 'SDXL', 'SDXL-Turbo']
    # images = [str(p) for p in Path('../../random_images_flux').glob('*.jpg')]  # old (400 images)
    images = [x['image'] for x in load_dataset('JonasLoos/imagenet_subset', split='train')]  # 500 images

    norms = {}
    for model in tqdm(models):
        print(f'Generating norms for {model}')
        sd = SD(model)
        reprs = sd.img2repr(images, sd.available_extract_positions, step=50, seed=42)
        norms[model] = {p: torch.stack([r.data[p].squeeze(0).norm(dim=0) for r in reprs], dim=0).mean(dim=0) for p in sd.available_extract_positions}

    pickle.dump(norms, open(path, 'wb'))



def norm_similarity_histogram(path: Path):

    # load model and dataset
    sd = SD('SD-1.5')
    dataset = [x['image'] for x in load_dataset('JonasLoos/imagenet_subset', split='train')]
    layers = sd.available_extract_positions

    # extract representations
    representations_noise_0 = sd.img2repr(dataset, extract_positions=layers, step=0)
    representations_noise_50 = sd.img2repr(dataset, extract_positions=layers, step=50)
    representations_noise_200 = sd.img2repr(dataset, extract_positions=layers, step=200)
    del sd, dataset

    # generate histograms
    n = 100
    rng = torch.Generator().manual_seed(42)
    histograms: dict[str, list[torch.Tensor|None]] = {pos: [None, None, None] for pos in layers}
    for pos in tqdm(layers):
        for i, representations in enumerate([representations_noise_0, representations_noise_50, representations_noise_200]):
            # calculate norms
            norms = torch.stack([r[pos].squeeze(0).flatten(start_dim=1).T for r in representations]).norm(dim=2).cpu()
            x_min, x_max = -0.5, 1.0
            y_min, y_max = norms.min().item(), norms.max().item()

            # calculate histogram
            # do this in a loop to avoid memory issues
            histogram = torch.zeros(n, n)
            for r1, n1 in zip(tqdm(representations), norms):
                for r2, n2 in zip(representations, norms):
                    if (torch.rand((1,), generator=rng) > 0.01 and pos in ['conv_in', 'up_blocks[2]', 'up_blocks[3]', 'conv_out']) or (torch.rand((1,), generator=rng) > 0.1 and pos in ['down_blocks[0]', 'up_blocks[1]']):
                        continue
                    # TODO: correct normalization of histogram
                    data = torch.stack([
                        r1.at(pos).cosine_similarity(r2.at(pos)).flatten().cpu(),
                        (n1[None,:]+n2[:,None]).flatten()/2
                    ], dim=1)
                    histogram += torch.histogramdd(data, bins=n, range=(x_min, x_max, y_min, y_max)).hist.T.flip(0)
            histograms[pos][i] = histogram

    # save histograms
    print(f'Saving {path}')
    torch.save(torch.stack([torch.stack(h) for h in histograms.values()]), path)

    # save y limits
    limits_path = path.with_name(path.stem + '_limits.pt')
    print(f'Saving {limits_path}')
    y_limits = torch.zeros(len(layers), 3, 2)
    for pos_i, pos in enumerate(layers):
        for i, representations in enumerate([representations_noise_0, representations_noise_50, representations_noise_200]):
            norms = torch.stack([r[pos].squeeze(0).flatten(start_dim=1).T for r in representations]).norm(dim=2).cpu()
            y_min, y_max = norms.min().item(), norms.max().item()
            y_limits[pos_i, i, 0] = y_min
            y_limits[pos_i, i, 1] = y_max
    torch.save(y_limits, limits_path)



def color_texture_bias_dense_correspondence_general(func: Callable[[np.ndarray, float], np.ndarray]):
    imagenet_subset = load_dataset("JonasLoos/imagenet_subset", split="train")
    images = [x['image'].convert('RGB') for x in tqdm(imagenet_subset, desc='Loading Images')]
    sd = SD()
    representations = []
    reference_representations = sd.img2repr(images, sd.available_extract_positions, 50, seed=0)  # different seed to avoid noise correlation
    for step in tqdm(np.linspace(0, 1, 10)):
        tmp_images = [func(np.array(img), step).astype(np.uint8) for img in images]
        representations.append(sd.img2repr(tmp_images, sd.available_extract_positions, 50, seed=42))
    accuracies = np.zeros((len(sd.available_extract_positions), len(representations), len(images)))
    for block_idx, block in enumerate(tqdm(sd.available_extract_positions)):
        for i in trange(len(images)):
            a = reference_representations[i].at(block).to('cuda')
            for int_idx, bs in enumerate(representations):
                b = bs[i].at(block).to('cuda')
                sim = a.cosine_similarity(b)
                n = sim.shape[0]
                accuracy = (sim.view(n*n, n*n).argmax(dim=0) == torch.arange(n*n, device='cuda')).float().mean().cpu()
                accuracies[block_idx, int_idx, i] = accuracy

    return accuracies


def color_bias_dense_correspondence_rgb_bgr(path: Path):
    degradation_fn = lambda arr, x: (x * arr[:,:,::-1] + (1-x) * arr).astype(np.uint8)
    accuracies = color_texture_bias_dense_correspondence_general(degradation_fn)
    np.save(path, accuracies)


def texture_bias_dense_correspondence_texture_overlay(path: Path):
    grass_arr = np.array(Image.open('images/grass.png').convert('RGB'))
    degradation_fn = lambda arr, x: (x/2 * grass_arr + (1-x/2) * arr).astype(np.uint8)
    accuracies = color_texture_bias_dense_correspondence_general(degradation_fn)
    np.save(path, accuracies)


def texture_bias_dense_correspondence_blur(path: Path):
    degradation_fn = lambda arr, x: np.array(Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=16 * x)))
    accuracies = color_texture_bias_dense_correspondence_general(degradation_fn)
    np.save(path, accuracies)


def texture_bias_dense_correspondence_noise(path: Path):
    degradation_fn = lambda arr, x: np.random.rand(*arr.shape) * 255 * x/2 + (1-x/2) * arr
    accuracies = color_texture_bias_dense_correspondence_general(degradation_fn)
    np.save(path, accuracies)



def color_bias_dense_correspondence_rgb_bgr_offset(path: Path):
    imagenet_subset = load_dataset("JonasLoos/imagenet_subset", split="train")
    images_big = [x['image'].convert('RGB').resize((768, 768)) for x in tqdm(imagenet_subset, desc='Loading Images')]
    images_upper_left = [x.crop((0, 0, 512, 512)) for x in images_big]
    images_lower_right = [x.crop((256, 256, 768, 768)) for x in images_big]
    sd = SD()
    representations = []
    reference_representations = sd.img2repr(images_upper_left, sd.available_extract_positions, 50, seed=0)  # different seed to avoid noise correlation
    for step in tqdm(np.linspace(0, 1, 10)):
        tmp_images = [(step * arr[:,:,::-1] + (1-step) * arr).astype(np.uint8) for img in images_lower_right for arr in [np.array(img)]]
        representations.append(sd.img2repr(tmp_images, sd.available_extract_positions, 50, seed=42))
    accuracies = np.zeros((len(sd.available_extract_positions), len(representations), len(images_big)))
    for block_idx, block in enumerate(tqdm(sd.available_extract_positions)):
        for i in trange(len(images_big)):
            a = reference_representations[i].at(block).to('cuda')
            for int_idx, bs in enumerate(representations):
                b = bs[i].at(block).to('cuda')
                sim = a.apply(lambda x: x[:,x.shape[1]//2:,x.shape[2]//2:]).cosine_similarity(b.apply(lambda x: x[:,:x.shape[1]//2,:x.shape[2]//2]))
                n = sim.shape[0]
                accuracy = (sim.view(n*n, n*n).argmax(dim=0) == torch.arange(n*n, device='cuda')).float().mean().cpu()
                accuracies[block_idx, int_idx, i] = accuracy

    np.save(path, accuracies)



def texture_bias_dense_correspondence_texture_overlay_offset(path: Path):
    imagenet_subset = load_dataset("JonasLoos/imagenet_subset", split="train")
    images_big = [x['image'].convert('RGB').resize((768, 768)) for x in tqdm(imagenet_subset, desc='Loading Images')]
    images_upper_left = [x.crop((0, 0, 512, 512)) for x in images_big]
    images_lower_right = [x.crop((256, 256, 768, 768)) for x in images_big]
    grass_arr = np.array(Image.open('images/grass.png').convert('RGB'))
    sd = SD()
    representations = []
    reference_representations = sd.img2repr(images_upper_left, sd.available_extract_positions, 50, seed=0)  # different seed to avoid noise correlation
    for step in tqdm(np.linspace(0, 0.5, 10)):
        tmp_images = [(step * grass_arr + (1-step) * arr).astype(np.uint8) for img in images_lower_right for arr in [np.array(img)]]
        representations.append(sd.img2repr(tmp_images, sd.available_extract_positions, 50, seed=42))
    accuracies = np.zeros((len(sd.available_extract_positions), len(representations), len(images_big)))
    for block_idx, block in enumerate(tqdm(sd.available_extract_positions)):
        for i in trange(len(images_big)):
            a = reference_representations[i].at(block).to('cuda')
            for int_idx, bs in enumerate(representations):
                b = bs[i].at(block).to('cuda')
                sim = a.apply(lambda x: x[:,x.shape[1]//2:,x.shape[2]//2:]).cosine_similarity(b.apply(lambda x: x[:,:x.shape[1]//2,:x.shape[2]//2]))
                n = sim.shape[0]
                accuracy = (sim.view(n*n, n*n).argmax(dim=0) == torch.arange(n*n, device='cuda')).float().mean().cpu()
                accuracies[block_idx, int_idx, i] = accuracy

    np.save(path, accuracies)



############################################################################
# run script
############################################################################

base_dir = Path('cached_data')
datafiles = {
    'representation_norms.pkl': generate_norms,
    'histograms_norm_cossim_sd15.pt': norm_similarity_histogram,  # also generates `..._limits.pt`
    'color_bias_dense_correspondence_rgb_bgr_accuracies.npy': color_bias_dense_correspondence_rgb_bgr,
    'texture_bias_dense_correspondence_texture_overlay_accuracies.npy': texture_bias_dense_correspondence_texture_overlay,
    'color_bias_dense_correspondence_rgb_bgr_accuracies_offset.npy': color_bias_dense_correspondence_rgb_bgr_offset,
    'texture_bias_dense_correspondence_texture_overlay_accuracies_offset.npy': texture_bias_dense_correspondence_texture_overlay_offset,
    'texture_bias_dense_correspondence_blur_accuracies.npy': texture_bias_dense_correspondence_blur,
    'texture_bias_dense_correspondence_noise_accuracies.npy': texture_bias_dense_correspondence_noise,
}


if __name__ == '__main__':
    argv = sys.argv[1:]
    if len(argv) > 1:
        print('Error: too many arguments')
        print('Usage: python generate_data.py [path]')
        exit(1)

    if len(argv) == 1 and argv[0] == 'help':
        print('Usage: python generate_data.py [path]')
        exit()

    if not base_dir.exists():
        print(f'Data cache directory {base_dir} does not exist. Are you in the assets folder?')
        exit(1)

    if len(argv) == 0:
        for filename, func in datafiles.items():
            path = base_dir / filename
            if path.exists():
                print(f'Skipping {filename} because it already exists')
            else:
                print()
                print('#'*50)
                print(f'Generating {filename}')
                print('#'*50)
                func(path)
                if not path.exists():
                    print(f'Failed to generate {filename}')
    else:
        path = Path(argv[0])
        if not path.is_relative_to(base_dir):
            print(f"Error: {path} is not in the base directory {base_dir}")
            exit(1)
        if path.name not in datafiles:
            print(f'Unknown datafile {path.name}')
            exit()
        if path.exists():
            print(f'overwriting {path.name}')
            sleep(2)  # wait to give the user a chance to abort

        datafiles[path.name](path)
        if not path.exists():
            print(f'Failed to generate {path.name}')
