# Dataset generation based on synthetic images for obstacle avoidance algorithms.
To get good results on neural network training a bunch of datasets correctly labelled is needed. This process is often very costly and tedious so this project shows a way of generating good a good quality dataset reducing the amount of effort needed by using some python algorithms.

In this Readme, you will find a brief description of this repository.
Later on, more information will be added since this project is still being carried out.


## Centromasas_Coll

This algorithm aims to locate all the non-collision points within an image. This way the objective point (ground truth) will point to a safe area where the drone will be able to fly safely.

The point will be calculated computing the centre of mass formula in all the non-collision points. The problem is that this algorithm has presents some singularities such as if the drone bumps into a tree in the middle of its trajectory the aim point will be on the tree, so it will be driving the drone directly into the tree. The following images show some singularity examples:


![Ejcm](images/Ejcm.png)


## Regression
This algorithm is born to solve the singularities of the former code. The operating principle is quite similar, but this time, instead of locating non-collision pixels, it will locate the position and main direction of all the collision pixels using a regression line. 
To place the aim point, a perpendicular line that passes through the centre of the image will be drawn. And finally, de distance from the intersection of the lines to the aim point will be inversely proportional to the number of collision pixels in the current image.


![Ejreg](images/Ejreg.png)


There are also some singularities, for example when the collision pixels are in both borders of the images. In this case, the regression line will be in the middle of the image and the distance to the aim point will lead the drone to the collision area. 


## Sectorizacion
This algorithm tries to overcome all the singularities seen before integrating the simplicity of the "Center of mass" algorithm.
Firstly the image is divided into four sub-images. Here the three sub-images with fewer collision pixels will be selected as they are deemed to be safer and the centre of mass will be computed in the three of them.
Finally, the aim point will be calculated using the centre of mass of the three images and the formula of the centre of mass again ponderating the position of each sub-centre of mass with the number of collision pixels in the sector.
Here is an image summing up how it works.


![Ejsect](images/Ejsect.png)


## pfm2png
This script decode the images extracted from Airsim in .pfm extension, into .png.

.pfm is an extension that ensures that the characters and the glyphs are correctly scaled. It is used in Airsim since depth is coded in decimal numbers within 0 and 1.
## Threading
In this project, a very simple prototype was developed based in Raspberry Pi. The hardware architecture is shown in the following image.
The configuration has been done following the UDEV rules as in this link. [UDEV Rules](http://raspberryjamberlin.de/zero360-part-2-connecting-via-otg-a-cluster-of-raspberry-pi-zeros-to-a-pi-3/ "UDEV Rules").
In this code, the ssh connection is solved using the threading library to ensure that both images are taken at the very same moment.
## Rotate
It is just a simple code to rotate the images a certain number of degrees.
## frame2vid
Convert a bunch of images into a video.
