from pathlib import Path

def f(sd_name, dataset_name, step=100, img_size=None):
    from sdhelper import SD
    from datasets import load_dataset, DatasetDict, Dataset
    import torch

    print('#'*50)
    print('sd_name:', sd_name)
    print('dataset_name:', dataset_name)
    print('step:', step)
    print('img_size:', img_size)
    print('#'*50)

    # Load stable diffusion model
    sd = SD(sd_name)

    # Load dataset
    dataset = load_dataset(dataset_name)
    assert isinstance(dataset, DatasetDict)
    if 'label' not in dataset['train'].features:
        if 'fine_label' in dataset['train'].features: dataset = dataset.rename_column('fine_label', 'label')
        else: raise ValueError('No label column found.')
    num_labels = len(dataset['train'].features['label'].names)
    # dataset = DatasetDict({k: Dataset.from_dict(v[:1000]) for k, v in dataset.items()})  # reduce dataset size
    # upscale images
    # if img_size is not None:
    #     img_key = next(iter({'img', 'image', 'feature'}.intersection(dataset['train'].column_names)))
    #     dataset = DatasetDict({k: Dataset.from_list([{img_key: x[img_key].resize((img_size, img_size)), 'label': x['label']} for x in v]) for k, v in dataset.items()})
    test_name = 'test'
    if test_name not in dataset:
        if 'valid' in dataset: test_name = 'valid'
        else: raise ValueError('No test or valid split found.')

    # Extract representations
    reprs = sd.img2repr(dataset, extract_positions=sd.available_extract_positions, step=step, resize=img_size, spatial_avg=True)
    assert isinstance(reprs, DatasetDict)

    # Classification
    results = {}
    for pos in sd.available_extract_positions:
        print('pos:', pos, end=' ')
        r = reprs['train'].with_format('torch')
        nn = torch.randn(r[pos][0].shape[-1], num_labels, requires_grad=True)
        optimizer = torch.optim.Adam([nn], lr=1e-3)

        # train
        for epoch in range(100):
            print('.', end='', flush=True)
            for batch in torch.utils.data.DataLoader(r, batch_size=128, shuffle=True):
                x, y = batch[pos][:,0,:], batch['label']
                y_hat = x @ nn
                loss = torch.nn.functional.cross_entropy(y_hat, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        print()

        # test
        accs = []
        for batch in torch.utils.data.DataLoader(reprs[test_name].with_format('torch'), batch_size=128):
            x, y = batch[pos][:,0,:], batch['label']
            y_hat = x @ nn
            acc = (y == y_hat.argmax(dim=1)).float().mean().item()
            accs.append(acc)
        print(f'{pos}: {sum(accs)/len(accs)}')
        results[pos] = sum(accs)/len(accs)

    # show accuracies
    print({'sd_name': sd_name, 'dataset_name': dataset_name, 'step': step, 'img_size': img_size})
    print(results)



if __name__ == '__main__':
    from hydrahelper import run_slurm_jobs
    run_slurm_jobs(
        config = {
            'job-name': 'classification',
            'partition': 'gpu-2d',
            'constraint': '80gb|40gb',
            'gpus-per-node': 1,
            'ntasks-per-node': 2,
            'mem': 30_000,
            'output': Path.home() / 'logs/job-%j-classification_upscaled.out',
        },
        python_cmd = 'apptainer run --nv ~/py311_container.sif python',
        function = f,
        args = [
            {'sd_name': sd_name, 'dataset_name': dataset_name, 'step': step, 'img_size': img_size}
            for sd_name in ['sd15', 'sd21', 'sd-turbo', 'sdxl', 'sdxl-turbo']
            for dataset_name in ['zh-plus/tiny-imagenet', 'cifar10', 'cifar100']
            for step in [50]
            for img_size in [256, 512]
        ]
    )
