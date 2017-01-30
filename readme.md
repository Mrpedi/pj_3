# Behavioral Cloning

## Predict Steering Angles for a Self Driving Car

### The Problem 
From a simulator provided by Udacity, ones should generate training data to train a neural network to predict steering angles.
Simulator's output returns image views from center, left and right of the car. In addition, an csv file containing:
* Steering Angles
* Break
* Throttle
* Speed

![left](https://s27.postimg.org/wvridfz73/left_sample.jpg) ![center](https://s27.postimg.org/yz1xl3yzz/center_sample.jpg) ![right](https://s27.postimg.org/71hpnnz73/right_sample.jpg) 

## Neural Network Model

The implemented model was inspired on NVIDIA's architecture as presented on this paper:
> https://arxiv.org/pdf/1604.07316v1.pdf

[![nvidia-model.png](https://s23.postimg.org/hsytusz1n/nvidia_model.png)](https://postimg.org/image/cubbg9v8n/)

### Conv Net

Although the model above has provided important details regarding its definition, it's not mentioned important information like hyper-parameters and non-linearity functions.

##### Implementation
In order to achieve a fast implementation of the model and quickly tweak architecture, [Keras](https:www.keras.io) has been used. Keras allows you to modify a model by just modifying A few lines of code.

##### Preprocessing
The number of training data is to high to be fully loaded in to memory, which demands images being loaded into batches to be normalized before the insertion into training pipeline. Preprocessing is carried out by *generator()* function, yielding data to training.  

##### Data Augmentation
No data augmentation was performed for this project.

##### Optimizer, Loss Function and Hyper Parameters

* Activation: RELU's after every layer
* Pooling Layers: After Every Convolutional Layer
    * Strides = 2x2
    
**Optimizer**: Adam optimizer - http://arxiv.org/abs/1412.6980v8
**Metric**: As the problem is characterized as a regressor, the correct metric is "mean squared error" which computes the mean of the squered difference between the predicted and correct value:

![MSE](https://wikimedia.org/api/rest_v1/media/math/render/svg/67b9ac7353c6a2710e35180238efe54faf4d9c15)

**Hyper Parameters**: 
* Learning Rate: 0.0001
* Batch Size: 128
* Number of Epochs: 2
* Dropout: 0.4

> model_dict = {'in_shape': (66, 200, 3),  # For tensorflow backend use shape as (row, cols, chan)
                  'nb_filter_1': 24, 'nb_row_1': 5, 'nb_col_1': 5, 'stride_1': (2, 2), 'z_1': 'relu', 'pad_1': 'valid',
                  'nb_filter_2': 36, 'nb_row_2': 5, 'nb_col_2': 5, 'stride_2': (2, 2), 'z_2': 'relu', 'pad_2': 'valid',
                  'nb_filter_3': 48, 'nb_row_3': 5, 'nb_col_3': 5, 'stride_3': (2, 2), 'z_3': 'relu', 'pad_3': 'valid',
                  'nb_filter_4': 64, 'nb_row_4': 2, 'nb_col_4': 2, 'stride_4': (1, 1), 'z_4': 'relu', 'pad_4': 'valid',
                  'nb_filter_5': 64, 'nb_row_5': 2, 'nb_col_5': 2, 'stride_5': (1, 1), 'z_5': 'relu', 'pad_5': 'valid',
                  'drop_p': 0.4}

#### Training, Validation and Testing
The inital dataset consisted of aproximatelly ten thousand samples, which was assessed as not sufficient for providing enough information to successfully train the neural net. To achieve a sufficient number of images, several sessions of running through cenarios one and two had been done, most importantly recording the session at specific points of the track to scape from collision path.

[![out_of_collision.gif](https://s29.postimg.org/dl84yqihj/out_of_collision.gif)](https://postimg.org/image/hhlguq3gz/)

**Data Set statistics:**
Data set has been splitted into 90% for training data and 10% for testing and validation:
* Train Dataset: 224.969
* Validation: 18747 
* Test Size: 6250
* Training time: 15 minutes with Geforce 1060 6GB GPU