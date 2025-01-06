WIP: unfinished, doesn't really work yet

Idea: restructure h-space in the early steps with the goal of aligning the image to some kind of segmentation mask

Steps:
* Select segmentation masks
* select h-space pixels that should belong together
* make grouped h-space pixels more similar using gradient descent
* regularize to prevent excessive changes
* generate image
