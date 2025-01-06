# Data Labeler

Webapp for labeling images with points of interest. Used for annotating high norm anomalies in the imagenet subset dataset.


## Usage

1. Put images into `images/` folder (or generate with `generate_norm_images.ipynb`)
2. Run `python app.py`
3. Open http://127.0.0.1:5001/
4. right click to label point of interest (click again to remove), left click to go to next image
5. repeat for all images
6. annotations are saved in the `images/` folder as `.json` files with the same name as the corresponding image.
7. combine annotations into a single file with `annotations_to_np.ipynb`
