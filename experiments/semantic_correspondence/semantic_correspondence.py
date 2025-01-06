from sdhelper import SD
import torch
import numpy as np
import datasets
from tqdm.autonotebook import tqdm, trange
import PIL.Image
import random
from collections import defaultdict
from typing import Literal, Callable, TypeAlias
from functools import partial
import traceback



##################################################
# Image tranformations
##################################################

def expand_and_resize(x: PIL.Image.Image, size = 960, border_pad=True):
    n, m = x.size
    s = max(n, m)
    r = PIL.Image.new('RGB', (s, s))
    r.paste(x, ((s-n)//2, (s-m)//2))
    if border_pad:
        # pad with border
        if n > m:
            r.paste(x.crop((0, 0, n, 1)).resize((n,(s-m)//2)), (0, 0))
            r.paste(x.crop((0, m-1, n, m)).resize((n,(s-m)//2)), (0, m+(s-m)//2))
        elif m > n:
            r.paste(x.crop((0, 0, 1, m)).resize(((s-n)//2,m)), (0, 0))
            r.paste(x.crop((n-1, 0, n, m)).resize(((s-n)//2,m)), (n+(s-n)//2, 0))
    return r.resize((size, size))

def expand_and_resize_keypoint(x, y, n, m, o, p):
    s = max(n, m)
    return (x + (s-n)//2) * o / s, (y + (s-m)//2) * p / s

def expand_resize_and_pad(x: PIL.Image.Image, size = 960, pad_size = 16):
    n, m = x.size
    o = max(n, m)
    s = size + 2 * pad_size
    r = PIL.Image.new('RGB', (s, s))
    r.paste(x.resize((int(size*n/o), int(size*m/o))), (pad_size+int(size*(1-n/o)/2), pad_size+int(size*(1-m/o)/2)))
    # corners
    cx = pad_size + int(size*(1-n/o)/2)
    cy = pad_size + int(size*(1-m/o)/2)
    r.paste(x.crop((0, 0, 1, 1)).resize((cx, cy)), (0, 0))
    r.paste(x.crop((0, m-1, 1, m)).resize((cx, s-cy)), (0, s-cy))
    r.paste(x.crop((n-1, 0, n, 1)).resize((s-cx, cy)), (s-cx, 0))
    r.paste(x.crop((n-1, m-1, n, m)).resize((s-cx, s-cy)), (s-cx, s-cy))
    # edges
    r.paste(x.crop((0, 0, 1, m)).resize((cx, s-2*cy)), (0, cy))
    r.paste(x.crop((n-1, 0, n, m)).resize((s-cx, s-2*cy)), (s-cx, cy))
    r.paste(x.crop((0, 0, n, 1)).resize((s-2*cx, cy)), (cx, 0))
    r.paste(x.crop((0, m-1, n, m)).resize((s-2*cx, s-cy)), (cx, s-cy))
    return r

def expand_resize_and_pad_keypoint(x, y, n, m, o, p, pad_size = 16):
    s = max(n, m)
    o -= 2 * pad_size
    p -= 2 * pad_size
    return (x + (s-n)//2) * o / s + pad_size, (y + (s-m)//2) * p / s + pad_size

def pad(x: PIL.Image.Image, pad_size = 16):
    n, m = x.size
    r = PIL.Image.new('RGB', (n + 2 * pad_size, m + 2 * pad_size))
    r.paste(x, (pad_size, pad_size))
    # corners
    r.paste(x.crop((0, 0, 1, 1)).resize((pad_size, pad_size)), (0, 0))
    r.paste(x.crop((0, m-1, 1, m)).resize((pad_size, pad_size)), (0, m+pad_size))
    r.paste(x.crop((n-1, 0, n, 1)).resize((pad_size, pad_size)), (n+pad_size, 0))
    r.paste(x.crop((n-1, m-1, n, m)).resize((pad_size, pad_size)), (n+pad_size, m+pad_size))
    # edges
    r.paste(x.crop((0, 0, 1, m)).resize((pad_size, m)), (0, pad_size))
    r.paste(x.crop((n-1, 0, n, m)).resize((pad_size, m)), (n+pad_size, pad_size))
    r.paste(x.crop((0, 0, n, 1)).resize((n, pad_size)), (pad_size, 0))
    r.paste(x.crop((0, m-1, n, m)).resize((n, pad_size)), (pad_size, m+pad_size))
    return r

def pad_keypoint(x, y, n, m, o, p, pad_size = 16):
    return x + pad_size, y + pad_size

def expand(x: PIL.Image.Image, size = 960):
    factor = size / min(x.size)
    return x.resize((int(x.size[0]*factor), int(x.size[1]*factor)))

def expand_keypoint(x, y, n, m, o, p):
    return x * o / n, y * p / m

transform_type: TypeAlias = Literal['expand_resize_and_pad', 'expand_and_resize', 'expand', None]

def get_transforms(name: transform_type, size=960):
    match name:
        case 'expand_resize_and_pad':
            return partial(expand_resize_and_pad, size=size), expand_resize_and_pad_keypoint
        case 'expand_and_resize':
            return partial(expand_and_resize, size=size), expand_and_resize_keypoint
        case 'pad':
            return pad, pad_keypoint
        case 'expand':
            return partial(expand, size=size), expand_keypoint
        case None:
            return lambda x, *args, **kwargs: x, lambda x, y, n, m, o, p: (x, y)
        case _:
            raise ValueError(f'Unknown transform name: {name}')



##################################################
# Semantic Correspondence
##################################################

@torch.no_grad()
def sc(
        sd: SD,
        dataset_pairs: datasets.Dataset,
        sample: dict,
        precomputed_reprs: list | None = None,
        extraction_step: int = 1,
        pos: list[str] = ['up_blocks[1]'],
        transform: Callable = lambda x: x,
        transform_keypoint: Callable = lambda x, y, *_: (x, y)):
    # load representations
    if precomputed_reprs is not None:
        repr1 = precomputed_reprs[sample['src_data_index']].at(pos)
        repr2 = precomputed_reprs[sample['trg_data_index']].at(pos)
    else:
        category = dataset_pairs.features['category'].names[sample['category']]
        repr1 = sd.img2repr(transform(sample['src_img']), extract_positions=pos, step=extraction_step, prompt=category)
        repr2 = sd.img2repr(transform(sample['trg_img']), extract_positions=pos, step=extraction_step, prompt=category)

    # concatenate representations
    repr1_full = repr1.concat().to(sd.device)
    repr2_full = repr2.concat().to(sd.device)

    # get images
    src_img = transform(sample['src_img'])
    trg_img = transform(sample['trg_img'])
    sn, sm = src_img.size
    tn, tm = trg_img.size
    assert len(sample['src_kps']) == len(sample['trg_kps'])

    # get bounding box
    sbb = np.array(sample['src_bndbox'])
    sbb[:2] = transform_keypoint(*sbb[:2], *sample['src_img'].size, sn, sm)
    sbb[2:] = transform_keypoint(*sbb[2:], *sample['src_img'].size, sn, sm)
    tbb = np.array(sample['trg_bndbox'])
    tbb[:2] = transform_keypoint(*tbb[:2], *sample['trg_img'].size, tn, tm)
    tbb[2:] = transform_keypoint(*tbb[2:], *sample['trg_img'].size, tn, tm)
    tbb_max = max(tbb[2] - tbb[0], tbb[3] - tbb[1])

    # solve semantic correspondence for each keypoint pair
    results = []
    for ([sx, sy],[tx,ty]) in zip(sample['src_kps'], sample['trg_kps']):

        # transform keypoints and bb
        sx, sy = transform_keypoint(sx, sy, *sample['src_img'].size, sn, sm)
        tx, ty = transform_keypoint(tx, ty, *sample['trg_img'].size, tn, tm)

        # calc similarities
        max_spatial1 = np.array(max(repr1[x].shape[-2:] for x in pos))
        point = repr1_full[:, int(sy/(sm/max_spatial1[-2])), int(sx/(sn/max_spatial1[-1])), None, None]
        similarities = torch.nn.functional.cosine_similarity(repr2_full, point, dim=0).cpu()  # cossim

        # similarities = (repr2_full - point).abs().mean(dim=0).cpu()  # MAE - doesn't seem to work well
        max_i = similarities.argmax().item()
        x_max = max_i % repr2_full.shape[-1]
        y_max = max_i // repr2_full.shape[-1]

        # calculate error distance -> PCK
        x_max_pixel = (x_max+.5) * tn / repr2_full.shape[-1]
        y_max_pixel = (y_max+.5) * tm / repr2_full.shape[-2]
        dist = ((x_max_pixel - tx)**2 + (y_max_pixel - ty)**2)**0.5
        dist_rel = dist / tbb_max
        correct = dist_rel <= 0.1
        results.append((correct, sx, sy, tx, ty, x_max_pixel, y_max_pixel, sn, sm, tn, tm, sample['category_id']))

    return results


def sc_calc_dataset(
        sd: SD,
        pos: list[str] = ['up_blocks[0]'],
        transform: transform_type = None,
        transform_size: int = 960,
        step: int = 1,
        use_prompt: bool = False,
        seed: int = 42,
        batch_size: int = 20,
    ):
    dataset_pairs: datasets.Dataset = datasets.load_dataset('0jl/SPair-71k', trust_remote_code=True, split='test')
    transform_img, transform_keypoint = get_transforms(transform, size=transform_size)

    # precalculate representations
    data_dataset = datasets.load_dataset('0jl/SPair-71k', 'data', trust_remote_code=True, split='train')
    img_data = [transform_img(x['img']) for x in tqdm(data_dataset, desc='transforming images')]
    prompts = [x['name'].split('/')[0] for x in data_dataset] if use_prompt else ''
    dataset_reprs = sd.img2repr(img_data, extract_positions=pos, step=step, prompt=prompts, output_device='cpu', batch_size=batch_size, seed=seed)

    results = []
    try:
        for sample in tqdm(dataset_pairs, desc='processing samples'):
            assert isinstance(sample, dict)
            results += sc(sd, dataset_pairs, sample, precomputed_reprs=dataset_reprs, extraction_step=step, pos=pos, transform=transform_img, transform_keypoint=transform_keypoint)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Full traceback:")
        traceback.print_exc()
    finally:
        return results


def random_hyper_opt(
        sd: SD,
    ):
    dataset_pairs: datasets.Dataset = datasets.load_dataset('0jl/SPair-71k', trust_remote_code=True, split='test')
    available_pos = [x for x in sd.available_extract_positions if any(f'{y}_block' in x for y in ['down', 'mid', 'up'])]
    runs = []
    try:
        for _ in trange(int(1e10)):
            # randomize hyperparameters
            t = ['expand_and_resize', 'expand', None][random.randint(0, 2)]
            p = random.sample(available_pos, random.randint(1, len(available_pos)))
            s = random.randint(1, 999)

            # run
            t1, t2 = get_transforms(t)
            sample = dataset_pairs[random.randint(0, len(dataset_pairs)-1)]
            pcks = sc(sd, dataset_pairs, sample, extraction_step=s, pos=p, transform=t1, transform_keypoint=t2)
            for pck in pcks:
                runs.append((dict(t=t,p=p,s=s), pck))
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)
    finally:
        print('Random Hyperparameter Optimization Results:')
        res = np.mean([x[1] for x in runs])
        print(f'PCK: {res:.2%} ({len(runs)} runs)')

        print('transforms:')
        for t in ['expand_and_resize', 'expand', None]:
            r = [x[1] for x in runs if x[0]["t"] == t]
            print(f'  {str(t)+":":<20} {np.mean(r):5.1%} ({np.mean(r)-res:+6.1%}) {f"({len(r)})":>7}')
        
        print('positions:')
        for p in available_pos:
            r = [x[1] for x in runs if p in x[0]["p"]]
            r_ = [x[1] for x in runs if p not in x[0]["p"]]
            print(f'  {p+":":<20} with: {np.mean(r):5.1%} ({np.mean(r)-res:+6.1%}) {f"({len(r)})":>7}, without: {np.mean(r_):.1%} ({np.mean(r_)-res:+6.1%}) {f"({len(r_)})":>7}')

        print('position combinations: ')
        pos_combinations = defaultdict(list)
        for x in runs:
            pos_combinations[tuple(sorted(x[0]["p"]))].append(x[1])
        for pos, pck in sorted(pos_combinations.items(), key=lambda x: -np.mean(x[1]))[:20]:
            print(f'  {str(pos):50}: {np.mean(pck):5.1%} ({np.mean(pck)-res:+6.1%}) {f"({len(pck)})":>7}')

        print('steps:')
        for s in range(0, 1000, 100):
            r = [x[1] for x in runs if s < x[0]["s"] <= x[0]["s"]+100]
            print(f'  {f"{s}-{s+100}":20} {np.mean(r):5.1%} ({np.mean(r)-res:+6.1%}) {f"({len(r)})":>7}')

        return runs

