from IPython.display import display, SVG
import hashlib
from pathlib import Path
import matplotlib.pyplot as plt
import inspect
import sys
import base64
from xml.etree import ElementTree as ET
from io import BytesIO
from PIL import Image as PILImage

cached_figures = set()


def cache_mpl_plot(plot_func=None, disabled=False, cache_path='assets/cached_plots', watch=[]):
    '''Cache a matplotlib plot, use as a decorator

    Args:
        func_or_filetype: The plotting function if the decorator is used without arguments. 
            If the decorator is used with arguments, this is used as the filetype.
        disabled: If True, the plot is not displayed, but also not purged from cache.
        cache_path: the path to the cache directory.
    '''
    plt.rcParams['svg.fonttype'] = 'none'  # make sure the text is selectable for svgs
    Path(cache_path).mkdir(parents=True, exist_ok=True)  # create the cache directory if it doesn't exist

    def decorator(plot_func):
        # Generate a unique filename based on the function name, arguments and source code
        func_name = plot_func.__name__
        func_code = inspect.getsource(plot_func).split('def ')[1]  # get source code without the decorator
        for x in watch:
            func_code += '\n' + inspect.getsource(x).split('def ')[1]
        func_hash = hashlib.md5(f'{func_name}{func_code}'.encode()).hexdigest()

        # check if the plot is already cached
        cache_path_full = None
        for tmp in Path(cache_path).glob(f'{func_name}_{func_hash}.svg'):
            cache_path_full = tmp
            break

        # create the plot if it doesn't exist
        if cache_path_full is None and not disabled:
            plot_func()
            cache_path_full = Path(cache_path) / f'{func_name}_{func_hash}.svg'
            svg_buffer = BytesIO()
            plt.savefig(svg_buffer, format='svg', dpi=300)
            svg_buffer.seek(0)
            save_compressed_svg(svg_buffer, cache_path_full)

            # clear vram if torch was used
            if 'torch' in sys.modules:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        plt.close()

        # add the plot to the cached figures, so it's not deleted when the cache is cleared
        if cache_path_full is not None:
            cached_figures.add(cache_path_full)

        if disabled:
            # display placeholder image when disabled
            plt.figure()
            plt.text(0.5, 0.5, f'Plot disabled\n({func_name})', ha='center', va='center')
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.show()
        else:
            # display the generated plot
            display(SVG(filename=str(cache_path_full)))

        # return dummy function that raises an error when called
        def fn(*args, **kwargs):
            raise Exception('Because this function is decorated with @cache_mpl_plot, it is already plotted and cannot be called directly.')
        return fn

    return decorator(plot_func) if callable(plot_func) else decorator



def reset_cached_figures():
    '''Reset the cached figures. Should be called at the beginning.'''
    while cached_figures:
        cached_figures.pop()


def clear_unused_cached_figures(cache_path='assets/cached_plots'):
    '''Clear all cached figures that are not used in the current notebook. Should be called at the end.'''
    for cached_figure in Path(cache_path).glob('plot_*_*.*'):
        if cached_figure not in cached_figures:
            print(f'deleting unused cached figure {cached_figure}', file=sys.stderr)
            cached_figure.unlink()



def save_compressed_svg(svg_file, output_file, quality=85):
    '''compress the images in an svg and save it to a new file'''
    ET.register_namespace('', 'http://www.w3.org/2000/svg')
    ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
    tree = ET.parse(svg_file)
    root = tree.getroot()

    for elem in root.findall('.//{http://www.w3.org/2000/svg}image'):
        href = elem.get('{http://www.w3.org/1999/xlink}href')
        if href and href.startswith('data:image'):
            _, b64data = href.split(',', 1)
            raw_data = base64.b64decode(b64data)

            with BytesIO(raw_data) as img_io:
                img = PILImage.open(img_io).convert('RGB')
                compressed_io = BytesIO()
                img.save(compressed_io, format='JPEG', quality=quality)
            
            new_data = base64.b64encode(compressed_io.getvalue()).decode()
            if len(new_data) < len(raw_data) * 0.9 - 10_000:  # only compress if the reduction is significant
                elem.set('{http://www.w3.org/1999/xlink}href', f'data:image/jpeg;base64,{new_data}')
    tree.write(output_file, encoding='utf-8', xml_declaration=True)



def display_figure(path: str, width: float, height: float):
    plt.figure(figsize=(width, height))
    plt.imshow(PILImage.open(path))
    plt.axis('off')
    plt.tight_layout()
    plt.show()



sd15_all_blocks = {
    'conv_in': [
        'conv_in',
    ],
    'down_blocks[0]': [
        'down_blocks[0].resnets[0]',
        'down_blocks[0].attentions[0]',
        'down_blocks[0].resnets[1]',
        'down_blocks[0].attentions[1]',
        'down_blocks[0].downsamplers[0]',
    ],
    'down_blocks[1]': [
        'down_blocks[1].resnets[0]',
        'down_blocks[1].attentions[0]',
        'down_blocks[1].resnets[1]',
        'down_blocks[1].attentions[1]',
        'down_blocks[1].downsamplers[0]',
    ],
    'down_blocks[2]': [
        'down_blocks[2].resnets[0]',
        'down_blocks[2].attentions[0]',
        'down_blocks[2].resnets[1]',
        'down_blocks[2].attentions[1]',
        'down_blocks[2].downsamplers[0]',
    ],
    'down_blocks[3]': [
        'down_blocks[3].resnets[0]',
        'down_blocks[3].resnets[1]',
    ],
    'mid_block': [
        'mid_block.resnets[0]',
        'mid_block.attentions[0]',
        'mid_block.resnets[1]',
    ],
    'up_blocks[0]': [
        'up_blocks[0].resnets[0]',
        'up_blocks[0].resnets[1]',
        'up_blocks[0].upsamplers[0]',
    ],
    'up_blocks[1]': [
        'up_blocks[1].resnets[0]',
        'up_blocks[1].attentions[0]',
        'up_blocks[1].resnets[1]',
        'up_blocks[1].attentions[1]',
        'up_blocks[1].resnets[2]',
        'up_blocks[1].attentions[2]',
        'up_blocks[1].upsamplers[0]',
    ],
    'up_blocks[2]': [
        'up_blocks[2].resnets[0]',
        'up_blocks[2].attentions[0]',
        'up_blocks[2].resnets[1]',
        'up_blocks[2].attentions[1]',
        'up_blocks[2].resnets[2]',
        'up_blocks[2].attentions[2]',
        'up_blocks[2].upsamplers[0]',
    ],
    'up_blocks[3]': [
        'up_blocks[3].resnets[0]',
        'up_blocks[3].attentions[0]',
        'up_blocks[3].resnets[1]',
        'up_blocks[3].attentions[1]',
        'up_blocks[3].resnets[2]',
        'up_blocks[3].attentions[2]',
    ],
    'conv_out': [
        'conv_out',
    ]
}
