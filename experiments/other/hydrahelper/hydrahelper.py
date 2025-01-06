from typing import Any, Callable
import pickle
import base64
import subprocess
import tempfile
import inspect


def run_slurm_jobs(config: dict[str,Any], python_cmd: str, function: Callable, args: list[dict[str,Any]]):
    """Run a function with different configurations in parallel using Slurm.

    Args:
        config: Slurm configuration dictionary.
        container: AppTainer container to run the function in.
        function: Function to run.
        args: List of argument dicts to pass to the given function. Must be pickle-able.

    Example:
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
    """
    # Make sure this function is called from the main file and not during the slurm job itself
    for frame_info in inspect.stack():
        if 'runpy' in frame_info.filename and 'run_path' in frame_info.function:
            print(f"WARNING: Please put the `run_slurm_jobs` call in `if __name__ == '__main__': ...`.")
            return

    # setup variables for the slurm file
    num_jobs = len(args)
    function_file_full_path = function.__code__.co_filename
    args_str = base64.b64encode(pickle.dumps(args)).decode('utf-8')

    # Setup the slurm file content
    file_content = [
        '#!/bin/bash',
        *[f'#SBATCH --{key}={value}' for key, value in config.items()],
        '',
        'echo "CURRENT TIME: $(date)"',
        'echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"',
        'echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"',
        'echo "SLURM_ARRAY_TASK_COUNT: $SLURM_ARRAY_TASK_COUNT"',
        'echo "SLURM_JOB_ID: $SLURM_JOB_ID"',
        '',
        f'ARGS={args_str}',
        f'{python_cmd} -c \'',
        '# Print python and gpu info',
        'import sys',
        'import torch as t',
        'print(f"python version: {sys.version}")',
        'print(f"cuda available: {t.cuda.is_available()}")',
        'print(f"cuda version: {t.version.cuda}")',
        'print(f"gpu: {t.cuda.is_available() and t.cuda.get_device_name(0)}")',
        'print("-"*100, flush=True);',
        '',
        '# Run the function with the given arguments',
        'import pickle, base64, runpy, os',
        'args = pickle.loads(base64.b64decode("\'$ARGS\'"))[\'$SLURM_ARRAY_TASK_ID\']',
        f'runpy.run_path("{function_file_full_path}")["{function.__name__}"](**args)',
        '\'',
    ]

    # Print the slurm file content for inspection
    newline = '\n'
    print(f'\nGenerated SLURM file: \n```\n{newline.join(l if len(l) < 200 else l[:200]+"..." for l in file_content)}\n```')

    # Ask for confirmation
    if input(f'Run {num_jobs} jobs? [y/N] ') != 'y':
        print('Aborted.')
        return
    
    # Submit the job to Slurm
    with tempfile.NamedTemporaryFile('w') as f:
        f.write('\n'.join(file_content))
        f.flush()
        retval = subprocess.run(['sbatch', '--array=0-'+str(num_jobs-1), f.name])
    
    if retval.returncode == 0:
        print('Remember to not change the used files while the jobs are running.')
    else:
        print('Error submitting job.')
