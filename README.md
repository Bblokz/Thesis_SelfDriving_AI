## AI Architectures Created for Autonomous Driving

### CarDrivingNetwork
The `CarDrivingNetwork` class serves as the foundation for a series of convolutional neural networks designed for predicting vehicle steering and throttle commands from image data. It contains a network architecture with convolutional layers followed by fully connected layers, using both image inputs and vehicle data to predict driving commands.

### DepthCarDrivingNetwork
Extending the base `CarDrivingNetwork`, the `DepthCarDrivingNetwork` modifies the first convolutional layer to accept input channels suitable for combined RGB and depth images, in this way we obtain the model's ability to learn from both visual and depth cues.

### PureDepthCarDrivingNetwork
The `PureDepthCarDrivingNetwork` adapts the `CarDrivingNetwork` to process pure depth information by adjusting the initial convolutional layer to accept single-channel depth images, focusing on depth-based learning for driving decisions.

### DoubleRGBCarDrivingNetwork
This model innovates on the `CarDrivingNetwork` by doubling the input channels in the first convolutional layer to process two consecutive RGB frames, with the aim to improve the prediction accuracy.

### DoubleRGBDepthCarDrivingNetwork
The `DoubleRGBDepthCarDrivingNetwork` expands on the `CarDrivingNetwork` architecture to ingest both RGB and depth data from two consecutive frames, providing a more vast, multidimensional dataset for the network to improve driving predictions.

## Module Summary

### Data Preparation Module
This module contains the reading and preparation of data for both training and testing phases of machine learning models. It includes functions to select data folders, prepare image datasets from specified paths, and handle both RGB and depth data for input into neural network models.

### Debugging Data Preparation Module
Designed to help in the visualization and debugging of data preparation, this module provides functions to display images alongside their vehicle data and labels. It supports debugging of both RGB and depth images, offering insights into the data feeding process.

### Network Module
This module defines the architecture of several neural network models for autonomous driving tasks, including models for RGB data, depth data, combined RGB and depth data, and models designed to process multiple consecutive frames. Each class extends a base neural network structure with specific convolutional layers to handle different input dimensions and data types.

### Testing Module
This module encompasses the functions required for evaluating neural network models against datasets to determine their performance in terms of accuracy, loss, and prediction precision. It dynamically loads the appropriate model based on the network type and calculates various metrics, including throttle and steering accuracy, sign prediction accuracy, and average absolute differences for both throttle and steering.

### Training Module
Focused on the training process of neural network models, this module handles the preparation and normalization of datasets, setting up the training loop, and optimizing the model parameters based on loss computation. It supports training for various model types, including RGB, depth, and combinations thereof, and ensures model parameters are saved upon training completion for future evaluation or continued training.
