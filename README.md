# ModelNet10-classifier
A classifier made using Pytorch to classify 3D shapes from the ModelNet10 dataset.  
The project includes data loading, preprocessing, local model training, and evaluation.
It's ultimately designed to display a comparison between different deep learning models and transformations for 3D shape classification using Tensorboard.

## Why this project?
I already worked on 2D classification on CIFAR, and I was curious about 3D classification from points clouds that could resemble what a LIDAR sensor would capture.  
I also wanted to create a modular codebase using Pytorch, allowing easy experimentation with different models and transformations to learn about optimizer and hyperparameter tuning.  
I'll also be able to work on implementing more models from known papers and get a good overview their processes, and maybe start one from scratch after.  
Finally, I wanted to practice using Tensorboard for visualizing training progress and results, since it's a already available tool and good to compare models performances.

## Tensorboard Visualization
To follow the training process, TensorBoard is used for visualization. Make sure you have TensorBoard installed.
#### Manual start:
If Tensorboard doesn't start automatically, run the following command in your terminal:

## Deep Learning Models
* PointNet (default)
* PointTransformer (in progress)
* DGCNN (in progress)
* PointNet++ (in progress)

## About the ModelNet10 Dataset
#### Download links:
* https://www.kaggle.com/datasets/balraj98/modelnet10-princeton-3d-object-dataset
* https://modelnet.cs.princeton.edu/  
#### Description:
The ModelNet10 dataset is widely used for benchmarking 3D shape classification algorithm containing a collection of 3D CAD models from 10 different categories: 
* bathtub
* bed 
* chair
* desk
* dresser
* monitor
* nightstand
* sofa
* table
* toilet  
The models are provided in OFF (Object File Format) files, a simple ASCII-based 3D mesh format that stores polygonal mesh data.  
#### OFF format structure:  
OFF  
_num_vertices num_faces num_edges_  
x1 y1 z1  
x2 y2 z2  
_..._  
xn yn zn  
n_vertices_face1 v1 v2 v3 ... [r g b a]  
n_vertices_face2 v1 v2 v3 ... [r g b a]  
_..._  

## Sampling
In order to be fed into the neural network, the 3D models need to be converted into point clouds.  
The points are sampled from the surface of the 3D models, to ensure uniformity in the input data, each 3D model is sampled to have exactly 1024 points (default value).  
We use a Farthest Point Sampling (FPS) algorithm from Open3D, which selects points that are as far apart from each other as possible, ensuring a good coverage of the model's surface.
### Note:
The sampling process is performed on-the-fly during training and evaluation, meaning that each time a model is loaded, a new set of points is sampled. This introduces variability in the input data, which can help improve the robustness of the trained models.  
However, this can be computationally expensive and would starve the GPU during training (the data pre-processing becomes the bottleneck).
To mitigate this, I will either precompute FPS sampled point clouds and save them to disk (using GPU acceleration if I can).  
A second possibility is to compute the next batch of sampled point clouds in a separate thread while the current batch is being processed by the GPU. This way, the CPU can prepare the next batch of data while the GPU is busy.
