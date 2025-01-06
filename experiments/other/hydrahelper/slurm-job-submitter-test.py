def f(x):
    print('hi')
    print(x)

if __name__ == '__main__':
    from hydrahelper import run_slurm_jobs
    from pathlib import Path
    run_slurm_jobs(
        config = {
            'job-name': 'test',
            'partition': 'gpu-test',
            'gpus-per-node': 1,
            'ntasks-per-node': 2,
            'output': Path.home() / 'logs/job-%j-test.out',
        },
        python_cmd = 'apptainer run --nv ~/py311_container.sif python',
        function = f,
        args = [
            {'x': 1},
            {'x': 2},
            {'x': 3},
        ]
    )
