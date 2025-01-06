WIP: doesn't work yet

Idea: backprop from image pixels to h-hspace to know which h-space regions are responsible for what

Questions
* Backprop from real image pixels or before autoencoder?
* Backprop from the last step or from each step separately?
* average over timesteps or have a separate map for each timestep?


Steps
* generate image
* use segment-anything to get segmentations in image space
* select segmentation mask to packprop
* sum over selected segmentation mask
* backprop to h-space
* visualize h-space relevance

Problems
* Keeping the gradients seems to take a huge amount of gpu memory -> only turbo models might be feasible when building the whole computation graph at once
