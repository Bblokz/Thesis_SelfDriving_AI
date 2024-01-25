import torch
import torch.nn as nn
import torch.nn.functional as F

# Module imports.
from Core.Constants import IMAGE_RESOLUTION


class CarDrivingNetwork(nn.Module):
    def __init__(self):
        super(CarDrivingNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=5, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=10, stride=2)

        # function to calculate size after conv layer
        def size_after_conv(size, kernel_size, stride, padding):
            return (size - kernel_size + 2*padding) // stride + 1

        # function to calculate size after max pool layer
        def size_after_maxpool(size, kernel_size, stride):
            return (size - kernel_size) // stride + 1

        # initial image size
        size = IMAGE_RESOLUTION

        # after conv1
        size = size_after_conv(size, 11, 4, 2)
        # after maxpool1
        size = size_after_maxpool(size, 3, 2)
        # after conv2
        size = size_after_conv(size, 5, 1, 2)
        # after maxpool2
        size = size_after_maxpool(size, 3, 2)
        # after conv3
        size = size_after_conv(size, 3, 1, 1)
        # after conv4
        size = size_after_conv(size, 3, 1, 1)
        # after conv5
        size = size_after_conv(size, 5, 1, 1)
        # after maxpool3
        size = size_after_maxpool(size, 10, 2)

        # the feature map dimensions are now (size x size x 256)
        # so the total number of features is:
        flat_output_size = size * size * 256

        # +2 for LastSteering and Speed.
        linear_input_size = flat_output_size + 2
        output_hidden1 = 1024 

        # Fully connected layers
        self.fc1 = nn.Linear(linear_input_size, output_hidden1)
        self.fc2 = nn.Linear(output_hidden1, 512)
        self.fc3 = nn.Linear(512, 256)
        
        # Output layers for throttle and steering.
        self.throttle_out = nn.Linear(256, 1)
        self.steering_out = nn.Linear(256, 1)
    
    def forward(self, image_data, vehicle_data):
        x = self.conv1(image_data)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.maxpool3(x)
        
        # Flatten the output for fully connected layers.
        x = x.view(x.size(0), -1)
        
        # Concatenate the vehicle data (LastSteering and Speed) with the flattened image data.
        x = torch.cat((x, vehicle_data), dim=1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        # Sigmoid for [0, 1] 
        throttle = torch.sigmoid(self.throttle_out(x)) 
        # Tanh for [-1, 1]
        steering = torch.tanh(self.steering_out(x))   
        
        return throttle, steering
    
    

class DepthCarDrivingNetwork(CarDrivingNetwork):
    def __init__(self):
        super(DepthCarDrivingNetwork, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=11, stride=4, padding=2)

class PureDepthCarDrivingNetwork(CarDrivingNetwork):
    def __init__(self):
        super(PureDepthCarDrivingNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
        
        
class DoubleRGBCarDrivingNetwork(CarDrivingNetwork):
    def __init__(self):
        super(DoubleRGBCarDrivingNetwork, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=11, stride=4, padding=2)
        
class DoubleRGBDepthCarDrivingNetwork(CarDrivingNetwork):
    def __init__(self):
        super(DoubleRGBDepthCarDrivingNetwork, self).__init__()
        self.conv1 = nn.Conv2d(8, 64, kernel_size=11, stride=4, padding=2)
        
class TrippleRGBCarDrivingNetwork(CarDrivingNetwork):
    def __init__(self):
        super(DoubleRGBDepthCarDrivingNetwork, self).__init__()
        self.conv1 = nn.Conv2d(9, 64, kernel_size=11, stride=4, padding=2)