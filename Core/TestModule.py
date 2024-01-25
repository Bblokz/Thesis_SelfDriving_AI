"""
Copyright (c) Bas Blokzijl Leiden University 2023-2024.
This module contains the functions used to test the model.
"""

import torch.nn
import tensorflow as tf
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
import os
import numpy as np
from enum import Enum

# Module imports
from Core.Constants import NetworkType, MODEL_PATH, MODEL_PATH_DEPTH, MODEL_PATH_PURE_DEPTH, MODEL_PATH_DOUBLE_RGB, MODEL_PATH_DOUBLE_RGB_DEPTH
from Core.NetworkModule import CarDrivingNetwork, DepthCarDrivingNetwork, PureDepthCarDrivingNetwork, DoubleRGBCarDrivingNetwork, DoubleRGBDepthCarDrivingNetwork
from Core.DataManipulationModule import prepare_depth_data, prepare_pure_depth_data, prepare_data, prepare_input_images

class TestType(Enum):
    Basic = 0
    Color = 1
    Different = 2

def test_model(image_dataset, vehicle_data, labels, model, device, network_type, throttle_accuracy_tolerance, steering_accuracy_tolerance):
    # Determine the correct model path based on network type
    if network_type == NetworkType.RGB:
        model_path = MODEL_PATH
    elif network_type == NetworkType.Depth:
        model_path = MODEL_PATH_DEPTH
    elif network_type == NetworkType.PureDepth:
        model_path = MODEL_PATH_PURE_DEPTH
    elif network_type == NetworkType.DoubleRGB:
        model_path = MODEL_PATH_DOUBLE_RGB
    elif network_type == NetworkType.DoubleRGBDepth:
        model_path = MODEL_PATH_DOUBLE_RGB_DEPTH

    throttle_percentage_tolerance = throttle_accuracy_tolerance * 100


    
    if not os.path.exists(model_path):
        print("No saved model parameters found. Please train the model first.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    # Set the model to evaluation mode
    model.eval()
    print(f"Model was loaded from {model_path}")
    
    # Convert TensorFlow dataset to numpy array.
    images_np = np.array([image.numpy() for image in image_dataset])
    # normalise the image values between 0 and 1.
    images_torch = torch.tensor(images_np).permute(0, 3, 1, 2).float() / 255.0
    
    # Convert TensorFlow dataset to numpy array.
    images_np = np.array([image.numpy() for image in image_dataset])
    # normalise the image values between 0 and 1.
    images_torch = torch.tensor(images_np).permute(0, 3, 1, 2).float() / 255.0
    
    images_torch = prepare_input_images(network_type, images_torch)

    vehicle_data_torch = torch.tensor(vehicle_data).float()
    labels_torch = torch.tensor(labels).float()
    # Create a TensorDataset from the tensors
    dataset = TensorDataset(images_torch, vehicle_data_torch, labels_torch)
    data_loader = DataLoader(dataset, batch_size=len(images_np), shuffle=True)
    
    # MSE
    loss_fn = torch.nn.MSELoss()
    
    total_loss = 0.0
    total_examples = 0
    correct_throttle_predictions = 0
    correct_steering_predictions = 0
    correct_steering_sign_predictions = 0
    total_abs_diff_throttle = 0.0
    total_abs_diff_steering = 0.0


    throttle_predictions_counter = Counter()
    steering_predictions_counter = Counter()

    with torch.no_grad(): 
        for batch_idx, (images, vehicle_data_batch, labels_batch) in enumerate(data_loader):
            images, vehicle_data_batch, labels_batch = images.to(device), vehicle_data_batch.to(device), labels_batch.to(device)

            # Forward pass: compute the model predictions
            throttle_pred, steering_pred = model(images, vehicle_data_batch)
            throttle_pred = throttle_pred.squeeze(-1)  
            steering_pred = steering_pred.squeeze(-1) 
            
            # Compute the loss.
            loss = loss_fn(throttle_pred, labels_batch[:, 0]) + loss_fn(steering_pred, labels_batch[:, 1])
            total_loss += loss.item() * images.size(0) 
            total_examples += labels_batch.size(0)  

            # Calculate accuracy based on the given tolerance
            for i in range(len(throttle_pred)):
                total_abs_diff_throttle += abs(throttle_pred[i].item() - labels_batch[i, 0].item())
                total_abs_diff_steering += abs(steering_pred[i].item() - labels_batch[i, 1].item())
                throttle_predictions_counter[round(throttle_pred[i].item(), 2)] += 1
                steering_predictions_counter[round(steering_pred[i].item(), 2)] += 1
                # Count how many times the correct sign was predicted, if zero steering is expected and zero is predicted we also count it as a correct sign prediction.
                if (steering_pred[i].item() * labels_batch[i, 1].item() >= 0) or (steering_pred[i].item() == 0 and labels_batch[i, 1].item() == 0):
                    correct_steering_sign_predictions += 1

               # Check if each prediction is within the acceptable range of actual values
                if abs(throttle_pred[i].item() - labels_batch[i, 0].item()) <= throttle_accuracy_tolerance:
                    correct_throttle_predictions += 1
                if abs(steering_pred[i].item() - labels_batch[i, 1].item()) <= steering_accuracy_tolerance:
                    correct_steering_predictions += 1

            avg_abs_steering = torch.mean(torch.abs(steering_pred)).item()
            avg_throttle = torch.mean(throttle_pred).item()


    # Calculate average loss
    average_loss = total_loss / total_examples

    avg_abs_diff_throttle = total_abs_diff_throttle / total_examples
    avg_abs_diff_steering = total_abs_diff_steering / total_examples

    # Find top 3 common throttle and steering predictions
    top_3_common_throttle = throttle_predictions_counter.most_common(3)
    top_3_common_steering = steering_predictions_counter.most_common(3)
    # Total predictions done.
    total_throttle_predictions = sum(throttle_predictions_counter.values())

    # Calculate the accuracy percentages
    throttle_accuracy_percentage = (correct_throttle_predictions / total_examples) * 100
    steering_accuracy_percentage = (correct_steering_predictions / total_examples) * 100
    steering_sign_accuracy_percentage = (correct_steering_sign_predictions / total_examples) * 100

    print(f'\n Total Loss: {total_loss}')
    print(f'Average Loss per Example: {average_loss}')
    print(f'\nAccuracy within {throttle_accuracy_tolerance} (within {throttle_percentage_tolerance}%) for throttle: {throttle_accuracy_percentage:.2f}% of examples')
    print(f'Accuracy within {steering_accuracy_tolerance} for steering: {steering_accuracy_percentage:.2f}% of examples')
    print(f'Accuracy for correct steering sign: {steering_sign_accuracy_percentage:.2f}% of examples')


    print(f'\nOut of {total_throttle_predictions} predictions top 3 is:')


    for i, (value, count) in enumerate(top_3_common_throttle, 1):
        print(f'Top {i} Throttle Prediction: {value} occurring {count} times')

    for i, (value, count) in enumerate(top_3_common_steering, 1):
        print(f'Top {i} Steering Prediction: {value} occurring {count} times')

    print(f'\nAverage Absolute Steering Value: {avg_abs_steering}')
    print(f'Average Throttle Value: {avg_throttle}')
    print(f'Average abs difference in throttle: {avg_abs_diff_throttle}')
    print(f'Average abs difference in steering: {avg_abs_steering}')

    return total_loss, average_loss, correct_throttle_predictions, correct_steering_predictions, correct_steering_sign_predictions, top_3_common_throttle, top_3_common_steering, avg_abs_diff_throttle, avg_abs_diff_steering


def test_data_sets(network_type, test_type):
    root_folder = './DATA'
    if test_type == TestType.Basic:
        run_folders = [f for f in os.listdir(root_folder) if f.startswith('Test') and not f.endswith('_Depth')]
    elif test_type == TestType.Color:
        run_folders = [f for f in os.listdir(root_folder) if f.startswith('C') and not f.endswith('_Depth')]
    elif test_type == TestType.Different:
        run_folders = [f for f in os.listdir(root_folder) if f.startswith('D') and not f.endswith('_Depth')]
   
    all_results = []
    total_examples = 0

    tolerances = [(0.05, 0.05), (0.1, 0.1)]

    # Create Statistics directory if it doesn't exist.
    stats_dir = './Statistics' 
    os.makedirs(stats_dir, exist_ok=True)

    if network_type == NetworkType.RGB:
        stats_file_name = 'regular.txt'
        network = CarDrivingNetwork()
    elif network_type == NetworkType.Depth:
        stats_file_name = 'depth.txt'
        network = DepthCarDrivingNetwork()
    elif network_type == NetworkType.PureDepth:
        stats_file_name = 'pureDepth.txt'
        network = PureDepthCarDrivingNetwork()
    elif network_type == NetworkType.DoubleRGB:
        stats_file_name = 'doubleRGB.txt'
        network = DoubleRGBCarDrivingNetwork()
    elif network_type == NetworkType.DoubleRGBDepth:
        stats_file_name = 'doubleRGBDepth.txt'
        network = DoubleRGBDepthCarDrivingNetwork()
    else:
        raise ValueError("Invalid network type")

    stats_file_path = os.path.join(stats_dir, stats_file_name)

    with open(stats_file_path, 'w') as stats_file:
        for throttle_tolerance, steering_tolerance in tolerances:
            throttle_accuracy_tolerance = throttle_tolerance
            steering_accuracy_tolerance = steering_tolerance
            throttle_percentage_tolerance = throttle_accuracy_tolerance * 100
            
            # Variables to accumulate total absolute differences
            total_abs_diff_throttle = 0.0
            total_abs_diff_steering = 0.0
            # Clear previous results if not the first run
            if throttle_tolerance != tolerances[0][0]:
                all_results.clear()
                total_examples = 0
                stats_file.write('\n\n')  # Separate the sections in the file
            stats_file.write(f'steering_accuracy_tolerance: {steering_accuracy_tolerance:.2f}\n')
            stats_file.write('RunFolder: ThrottleAccuracy% SteeringAccuracy% SteeringSignAccuracy% Loss AvgAbsDiffThrottle AvgAbsDiffSteering\n')
            for run_folder in run_folders:
                folder_path = os.path.join(root_folder, run_folder)
                print(f'Testing data from {folder_path}')
                
                if network_type == NetworkType.RGB or network_type == NetworkType.DoubleRGB:
                    image_dataset, vehicle_data, labels = prepare_data(folder_path)
                elif network_type == NetworkType.Depth or network_type == NetworkType.DoubleRGBDepth:
                    image_dataset, vehicle_data, labels = prepare_depth_data(folder_path)
                elif network_type == NetworkType.PureDepth:
                    image_dataset, vehicle_data, labels = prepare_pure_depth_data(folder_path)

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                network.to(device)
                results = test_model(image_dataset, vehicle_data, labels, network, device, network_type, throttle_accuracy_tolerance, steering_accuracy_tolerance)
                total_examples += labels.shape[0]
                all_results.append(results)

                # Write individual run results
                throttle_accuracy_percentage = (results[2] / labels.shape[0]) * 100
                steering_accuracy_percentage = (results[3] / labels.shape[0]) * 100
                steering_sign_accuracy_percentage = (results[4] / labels.shape[0]) * 100
                total_abs_diff_throttle += results[7] * labels.shape[0]
                total_abs_diff_steering += results[8] * labels.shape[0]
                stats_file.write(f'{run_folder}: {throttle_accuracy_percentage:.2f} {steering_accuracy_percentage:.2f} {steering_sign_accuracy_percentage:.2f} {results[1]:.4f} {results[7]:.4f} {results[8]:.4f}\n')

            # Calculate overall statistics
            total_loss = sum(result[0] for result in all_results)
            average_loss = total_loss / sum(len(result) for result in all_results)
            correct_throttle_predictions = sum(result[2] for result in all_results)
            correct_steering_predictions = sum(result[3] for result in all_results)
            correct_steering_sign_predictions = sum(result[4] for result in all_results)
            throttle_accuracy_percentage = (correct_throttle_predictions / total_examples) * 100
            steering_accuracy_percentage = (correct_steering_predictions / total_examples) * 100
            steering_sign_accuracy_percentage = (correct_steering_sign_predictions / total_examples) * 100

            # Calculate average absolute differences over all datasets
            avg_abs_diff_throttle = total_abs_diff_throttle / total_examples
            avg_abs_diff_steering = total_abs_diff_steering / total_examples

            # Write overall statistics to file
            stats_file.write('\nRunFolder OVERALL: ThrottleAccuracy% SteeringAccuracy% SteeringSignAccuracy% Loss TotalExamples AvgAbsDiffThrottle AvgAbsDiffSteering\n')
            stats_file.write(f'OVERALL: {throttle_accuracy_percentage:.2f} {steering_accuracy_percentage:.2f} {steering_sign_accuracy_percentage:.2f} {average_loss:.4f} {total_examples} {avg_abs_diff_throttle:.4f} {avg_abs_diff_steering:.4f}\n')

    print(f"Stats file path: {stats_file_path}")

    return all_results

