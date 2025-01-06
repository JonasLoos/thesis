# Hydra Helper

A helper package for working with the hydra cluster, e.g. for submitting slurm jobs.


## Installation

Only install on the hydra cluster.

```bash
pip install -e .
```

## Usage

```python
def f(x):
    print('hi')
    print(x)

if __name__ == '__main__':
    from hydrahelper import run_slurm_jobs
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
```


## Hydra Notes

docs: https://git.tu-berlin.de/ml-group/hydra/documentation/


### Active jobs

```bash
squeue -u $USER
```


### Update apptainer image

```bash
srun --partition=cpu-2h --pty bash
apptainer build py311_container.sif py311_container\ copy.def
```
