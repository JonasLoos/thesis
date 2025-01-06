# Stable Diffusion Unet Architectures

This directory contains the unet architectures extracted by the following code:

```python
from sdhelper import SD
from pathlib import Path


models = [
    'SD1.5',
    'SD2.1',
    'SD-Turbo',
    'SDXL-Turbo',
    'SDXL-Lightning-1step',
    # 'SDXL-Lightning-2step',
    'SDXL-Lightning-4step',
    # 'SDXL-Lightning-8step',
]


save_path = Path('unet-architecture')
save_path.mkdir(exist_ok=True)
for model in models:
    print(model)
    sd = SD(model)
    arch = str(sd.pipeline.unet)
    with open(save_path / f'{model}.txt', 'w') as f:
        f.write(arch)
```

In general the structure of stable diffusion unets looks like this:

* `conv_in`: convolutional layer
* `down_blocks`: 3-4 blocks each containing resnet and partly transformer blocks followed by a downsampling layer
* `mid_block`: block containing resnet and transformer blocks
* `up_blocks`: 3-4 blocks each containing resnet and partly transformer blocks followed by an upsampling layer
* `conv_out`: convolutional layer
