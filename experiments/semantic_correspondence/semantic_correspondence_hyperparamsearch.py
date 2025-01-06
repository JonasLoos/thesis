from pathlib import Path
import importlib

def f(sd_name, **kwargs):
    print(kwargs)
    import importlib
    from sdhelper import SD
    import numpy as np
    sc = importlib.import_module('semantic_correspondence')
    sd = SD(sd_name)
    results = sc.sc_calc_dataset(sd, **kwargs)
    result_data = np.array(results)
    path = Path.home() / f'sc-hyper-results/step50_vary_all_else/{sd.model_name}-{"+".join(kwargs["pos"])}-{str(kwargs["transform"])}-{kwargs["transform_size"]}-{kwargs["step"]}.npy'
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, result_data)


if __name__ == '__main__':
    # assert Path.cwd().name == 'semantic_correspondence', 'Please run this script from the semantic_correspondence folder.'
    hydrahelper = importlib.import_module('hydrahelper')

    base_path = Path.home() / f'sc-hyper-results/step50_vary_all_else/'
    existing_results = [str(p.stem) for p in base_path.glob('*.npy')]

    hydrahelper.run_slurm_jobs(
        config = {
            'job-name': 'sc-hyper',
            'partition': 'gpu-2h',
            'gpus-per-node': 1,
            'ntasks-per-node': 2,
            'mem': 100_000,
            'output': Path.home() / 'logs/job-%j-sc-hyper.out',
            'constraint': "h100|80gb"  # "h100|80gb|40gb|6000|3090"
        },
        python_cmd = 'apptainer run --nv ~/py311_container.sif python',
        function = f,
        args = [
            {'sd_name': sd_name, 'pos': [pos], 'transform': transform, 'transform_size': transform_size, 'step': step}
            for sd_name in ['SD1.5', 'SD2.1', 'SD-Turbo', 'SDXL', 'SDXL-Turbo']
            # for pos in sorted([
            #     'mid_block',
            #     *[f'{d}_blocks[{i}]' for i in range(4) for d in ['up', 'down']],
            #     'mid_block.attentions[0]',
            #     'mid_block.resnets[0]',
            #     'mid_block.resnets[1]',
            #     *[f'{d}_blocks[{i}].{t}[{j}]'
            #       for d, tmp in [
            #           ('down', [(2,2), (2,2), (2,2), (0,2)]),
            #           ('up', [(0,3), (3,3), (3,3), (3,3)]),
            #       ]
            #       for i, (a_len, r_len) in enumerate(tmp)
            #       for t, j in zip(['attentions']*a_len + ['resnets']*r_len, [*range(a_len), *range(r_len)])
            #     ],
            # ])
            # for pos in ['conv_in','down_blocks[0]','down_blocks[1]','down_blocks[2]','down_blocks[3]','mid_block','up_blocks[0]','up_blocks[1]','up_blocks[2]','up_blocks[3]','conv_out']
            for pos in ['conv_in','down_blocks[0]','down_blocks[1]','down_blocks[2]','mid_block','up_blocks[0]','up_blocks[1]','up_blocks[2]','conv_out'] + ([] if 'SDXL' in sd_name else ['down_blocks[3]', 'up_blocks[3]'])
            for transform in ['expand_and_resize']  # [None, 'pad', 'expand', 'expand_and_resize', 'expand_resize_and_pad']
            for transform_size in [256, 512, 768, 1024]
            for step in [50]  # [0, 10, 25, 50, 75, 100, 150, 200, 300, 500, 800]
            if f'{sd_name}-{pos}-{transform}-{transform_size}-{step}' not in existing_results
        ]
    )
