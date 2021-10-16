# Dataset generation based on synthetic images for obstacle avoidance algorithms.
Machine learning is able to provide autonomous vehicles with some intelligence for their navigation functions. One of the main functions that a UAV must be able to solve is object avoidance, which will be the focus of this work. 
For the proper training of any neural network, a large amount of information is required. In this case, images of various landscapes to be captured from the moving vehicle. Then, in order to safeguard the integrity of the drone, virtual environments used in video games will be of great help to generate synthetic images. This will make it possible to obtain image sequences also in environments where it would be unthinkable to fly a drone, such as, for example, the lunar surface or any unknown and inaccessible place today. In particular, the Unreal Engine graphics engine has been used, in combination with the Airsim plug-in, which embeds the drone's movement within the virtual landscape and allows the image acquisition format to be configured.
To generate the training data for the neural network, we start from the previous images whose pixels will encode the depth and, therefore, an indicator of the probability of collision will be obtained. This provides a solution to one of the main problems in machine learning algorithms known as "transfer learning". In many applications, neural networks learn very specific characteristics of elements such as textures, colours, compositions, which subsequently reduce the versatility of the neural network. In this case, the image is encoded in greyscale so that the neural network learns to differentiate the geometries that appear, and desensitising its response to colours or textures.
In order to automate the generation of the ground-truth, three segmentation algorithms have been developed, which generate as output the optimal point (X-Y coordinates) or the point with the lowest probability of collision. The three algorithms (Centre of Mass, Regression, Sectorisation) use physical and statistical formulas such as the centre of mass formula, or regression line calculations used for the identification of the position and direction of the collision pixels. Finally, the results of each of them will be compared in a ResNet-8 neural network that as output yields a target point within the area viewed from the UAV where the lowest probability of collision is estimated.


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

##Results
##Reliability
Although the algorithms yield an optimal target point with the lowest possible collision probability, it should be noted that in some cases the calculation is complex due to the large number of obstacles or their position in space. Therefore, once all the points have been calculated, the probability of collision of the specific area where the drone is being directed must be analysed.
The point calculated by the algorithms will be used as the centre of a circle of fixed radius r which will be 70 pixels.
![Ejsect](images/Reliability.png)

Translated with www.DeepL.com/Translator (free version)
##Estimated values vs. Groundtruth
![Ejsect](images/Errorcomparation.png)
Error distribution analysis between estimated values and round truth. 
![Ejsect](images/Errordistribution.png)
#Error:RMSE
The root mean squared error between the values for the target point predicted by the neural network and the values for the actual target point used for training will also be analysed. In this way and with a single coefficient, it will be possible to know how good the learning of the neural network can be.

![Ejsect](images/Errorcomparation.png)
