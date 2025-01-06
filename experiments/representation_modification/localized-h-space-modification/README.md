WIP: doesn't work yet

Idea: Explore h-space modification with localized directions that are generated from different spatial locations and applied only where relevant.

Steps
* generate many images of two groups, one base group and one group with a modification
* somehow find regions in h-space that are similar in one image group and similar in the other, but different between the two groups
* average the directions from base to modification in these regions
* apply these directions to new images at pixels that are similar to the regions in the base group
