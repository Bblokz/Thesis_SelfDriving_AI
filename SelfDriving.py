"""
Copyright (c) Bas Blokzijl Leiden University 2023-2024.
This file contains the main function for the self-driving car project.
In here we created different models and train or test them using the Core modules.
"""


import torch

# Module imports
from Core.Constants import NetworkType
from Core.DataManipulationModule import prepare_data_sets
from Core.DebugModule import debug_plot
from Core.TrainModule import train_model
from Core.TestModule import test_model, test_data_sets, TestType
from Core.NetworkModule import CarDrivingNetwork, DepthCarDrivingNetwork, PureDepthCarDrivingNetwork, DoubleRGBCarDrivingNetwork, DoubleRGBDepthCarDrivingNetwork

        

def PrintModelAndDefineDevice(network):
    print(network)
    total_params = count_parameters(network)
    print(f'Total trainable parameters: {total_params}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Device name:", torch.cuda.get_device_name(device))
    return device

def PickModleTrainingOrTesting(networkType, network, device):
    action = input("Do you want to train the model or test it? (train/test): ").lower()
    image_dataset, vehicle_data, labels = prepare_data_sets(networkType)
    debug = input("Do you want to debug the data? (yes/no): ").lower() == 'yes'
    if debug:
        debug_plot(image_dataset, vehicle_data, labels)
    if action == "train":
        train_model(image_dataset, vehicle_data, labels, network, device, networkType)
    elif action == "test":
        test_model(image_dataset, vehicle_data, labels, network, device, networkType, 0.1, 0.1)
    else:
        print("Invalid option selected.")

# This model takes the current and previous frame as input.
def InitialiseDoubleRGBModel():
    network = DoubleRGBCarDrivingNetwork()
    device = PrintModelAndDefineDevice(network)
    PickModleTrainingOrTesting(NetworkType.DoubleRGB, network, device)
    
# This model takes the depth and previous depth frame as input as well as the rgb values of both frames.
def InitialiseDoubleRGBDepthModel():
    network = DoubleRGBDepthCarDrivingNetwork()
    device = PrintModelAndDefineDevice(network)
    PickModleTrainingOrTesting(NetworkType.DoubleRGBDepth, network, device)
    
def InitialiseRGBModel():
    network = CarDrivingNetwork()
    device = PrintModelAndDefineDevice(network)
    PickModleTrainingOrTesting(NetworkType.RGB, network, device)

def InitialiseDepthModel():
    network = DepthCarDrivingNetwork()
    device = PrintModelAndDefineDevice(network)
    PickModleTrainingOrTesting(NetworkType.Depth, network, device)

def InitialisePureDepthModel():
    network = PureDepthCarDrivingNetwork()
    device = PrintModelAndDefineDevice(network)
    PickModleTrainingOrTesting(NetworkType.PureDepth, network, device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def main():
    while True:
        print("Choose an option to run:")
        print("1. RGB Model")
        print("2. Depth Model")
        print("3. Pure Depth Model")
        print("4. Full Test Depth")
        print("5. Full Test RGB")
        print("6. Full Test Pure Depth")
        print("7. Color Test Detph")
        print("8. Color Test RGB")
        print("9 Color Test Pure Detph")
        print("a. Double RGB Model")
        print("b. Double RGB Test")
        print("c. Double RGB Depth Model")
        print("d. Double RGB Depth Test")
        print("0. Exit")
        model_choice = input("Enter your choice (0-9): ")

        if model_choice == "1":
            InitialiseRGBModel()
        elif model_choice == "2":
            InitialiseDepthModel()
        elif model_choice == "3":
            InitialisePureDepthModel()
        elif model_choice == "4":
            test_data_sets(NetworkType.Depth, TestType.Basic)
        elif model_choice == "5":
            test_data_sets(NetworkType.RGB, TestType.Basic)
        elif model_choice == "6":
            test_data_sets(NetworkType.PureDepth, TestType.Basic)
        elif model_choice == "7":
            test_data_sets(NetworkType.Depth, TestType.Color)
        elif model_choice == "8":
            test_data_sets(NetworkType.RGB, TestType.Color)
        elif model_choice == "9":
            test_data_sets(NetworkType.PureDepth, TestType.Color)
        elif model_choice == "a":
            InitialiseDoubleRGBModel()
        elif model_choice == "b":
            test_data_sets(NetworkType.DoubleRGB, TestType.Basic)
        elif model_choice == "c":
            InitialiseDoubleRGBDepthModel()
        elif model_choice == "d":
            test_data_sets(NetworkType.DoubleRGBDepth, TestType.Basic)
        elif model_choice == "0":
            break 
        else:
            print("Invalid option selected. Please try again.")

if __name__ == "__main__":
    main()


