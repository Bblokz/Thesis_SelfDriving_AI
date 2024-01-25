from enum import Enum

IMAGE_RESOLUTION = 256
MODEL_PATH = './SelfDrivingModel/model_parameters.pth'
MODEL_PATH_DEPTH = './SelfDrivingModelDepth/model_parameters.pth'
MODEL_PATH_PURE_DEPTH = './SelfDrivingModelPureDepth/model_parameters.pth'
MODEL_PATH_DOUBLE_RGB = './SelfDrivingModelDoubleRGB/model_parameters.pth'
MODEL_PATH_DOUBLE_RGB_DEPTH = './SelfDrivingModelDoubleRGBDepth/model_parameters.pth'

class NetworkType(Enum):
    RGB = 0
    Depth = 1
    PureDepth = 2
    DoubleRGB = 3
    DoubleRGBDepth = 4