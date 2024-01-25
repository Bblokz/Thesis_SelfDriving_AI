"""
Copyright (c) Bas Blokzijl, Leiden University 2023-2024.

Module for reading and preparing data for training and testing.
"""	

import os
import tensorflow as tf
import pandas as pd
import numpy as np
import torch
# Module imports.
from Core.Constants import NetworkType

def choose_data_folder():
    """
    @brief Lists available data folders and allows user to choose one for data preparation.
    @return The path of the chosen data folder as a string.
    @note Assumes data folders are prefixed with 'Run' and located within a 'DATA' directory
    relative to the current working directory.
    """
    root_folder = './DATA'
    run_folders = [f for f in os.listdir(root_folder) if f.startswith('Run') and not f.endswith('_Depth')]
    for idx, folder in enumerate(run_folders):
        print(f'{idx+1}. {folder}')
    choice = int(input("Enter the number of the data folder you want to choose: "))
    chosen_folder = os.path.join(root_folder, run_folders[choice-1])
    return chosen_folder


def prepare_data(folder_path):
    """
    @brief Prepares image and CSV data from the specified folder for model input and training.
    @param folder_path: The path to the data folder containing images and a CSV file.
    @return A tuple containing:
    - image_dataset: A TensorFlow dataset object containing the images.
    - vehicle_data: A NumPy array containing the LastSteering and Speed data.
    - labels: A NumPy array containing the Throttle and Steering labels.
    
    @note Assumes images are named in the format image_<frame_count>.png and the CSV file
    contains columns Count, Throttle, Steering, LastSteering, and Speed.
    """
    csv_file_path = os.path.join(folder_path, 'vehicle_data.csv')
    csv_data = pd.read_csv(csv_file_path)
    # cast to int as we get errors when we try to use float values. (46.0 != 46)
    image_paths = [os.path.join(folder_path, f'image_number_{int(row["Count"])}.png') for index, row in csv_data.iterrows()]
    # Create dataset where each element is filled by a file path to an image.
    # Take file path x and read it, returning a tensor of type tf.string which takes the raw bytes of the images.
    image_dataset = tf.data.Dataset.from_tensor_slices(image_paths).map(lambda x: tf.image.decode_png(tf.io.read_file(x)))
    vehicle_data = csv_data[['LastSteering', 'Speed']].values
    labels = csv_data[['Throttle', 'Steering']].values
    return image_dataset, vehicle_data, labels


def prepare_depth_data(folder_path):
    depth_folder_path = folder_path + '_Depth'
    csv_file_path = os.path.join(folder_path, 'vehicle_data.csv')
    csv_data = pd.read_csv(csv_file_path)

    image_paths = [os.path.join(folder_path, f'image_number_{int(row["Count"])}.png') for index, row in csv_data.iterrows()]
    depth_image_paths = [os.path.join(depth_folder_path, f'image_number_{int(row["Count"])}.png') for index, row in csv_data.iterrows()]

    # Load RGB and depth images, concatenate along the channel dimension
    def load_and_combine_images(rgb_path, depth_path):
        rgb_image = tf.image.decode_png(tf.io.read_file(rgb_path))
        depth_image = tf.image.decode_png(tf.io.read_file(depth_path), channels=1)  
        combined_image = tf.concat([rgb_image, depth_image], axis=-1)
        return combined_image

    image_dataset = tf.data.Dataset.from_tensor_slices((image_paths, depth_image_paths)).map(load_and_combine_images)
    vehicle_data = csv_data[['LastSteering', 'Speed']].values

    labels = csv_data[['Throttle', 'Steering']].values

    return image_dataset, vehicle_data, labels

def prepare_pure_depth_data(folder_path):
    depth_folder_path = folder_path + '_Depth'
    csv_file_path = os.path.join(folder_path, 'vehicle_data.csv')
    csv_data = pd.read_csv(csv_file_path)

    depth_image_paths = [os.path.join(depth_folder_path, f'image_number_{int(row["Count"])}.png') for index, row in csv_data.iterrows()]

    def load_depth_images(depth_path):
        depth_image = tf.image.decode_png(tf.io.read_file(depth_path), channels=1) 
        return depth_image

    image_dataset = tf.data.Dataset.from_tensor_slices(depth_image_paths).map(load_depth_images)
    vehicle_data = csv_data[['LastSteering', 'Speed']].values
    labels = csv_data[['Throttle', 'Steering']].values

    return image_dataset, vehicle_data, labels


def prepare_data_sets(network_type):
    image_datasets = []
    vehicle_datasets = []
    labelsets = []
    while True:
        print(f"Current total examples: {sum([len(labels) for labels in labelsets])}")
        print("1: Add more data.")
        print("2: Add all run data.")
        print("3: Exit.")
        choice = int(input("Enter your choice: "))
        if choice == 1:
            chosen_folder = choose_data_folder()
            if network_type == NetworkType.RGB or network_type == NetworkType.DoubleRGB:
                image_dataset, vehicle_data, labels = prepare_data(chosen_folder)
            elif network_type == NetworkType.Depth or network_type == NetworkType.DoubleRGBDepth:
                image_dataset, vehicle_data, labels = prepare_depth_data(chosen_folder)
            elif network_type == NetworkType.PureDepth:
                image_dataset, vehicle_data, labels = prepare_pure_depth_data(chosen_folder)
            image_datasets.append(image_dataset)
            vehicle_datasets.append(vehicle_data)
            labelsets.append(labels)
        elif choice == 2:
            root_folder = './DATA'
            run_folders = [f for f in os.listdir(root_folder) if f.startswith('Run') and not f.endswith('_Depth')]
            for run_folder in run_folders:
                folder_path = os.path.join(root_folder, run_folder)
                if network_type == NetworkType.RGB or network_type == NetworkType.DoubleRGB:
                    image_dataset, vehicle_data, labels = prepare_data(folder_path)
                elif network_type == NetworkType.Depth or network_type == NetworkType.DoubleRGBDepth:
                    image_dataset, vehicle_data, labels = prepare_depth_data(folder_path)
                elif network_type == NetworkType.PureDepth:
                    image_dataset, vehicle_data, labels = prepare_pure_depth_data(folder_path)
                image_datasets.append(image_dataset)
                vehicle_datasets.append(vehicle_data)
                labelsets.append(labels)
        elif choice == 3:
            # Combine all datasets into single datasets
            combined_image_dataset = image_datasets[0]
            for dataset in image_datasets[1:]:
                combined_image_dataset = combined_image_dataset.concatenate(dataset)
            combined_vehicle_data = np.vstack(vehicle_datasets)
            combined_labels = np.vstack(labelsets)
            return combined_image_dataset, combined_vehicle_data, combined_labels
        else:
            print("Invalid choice. Please try again.")


def prepare_input_images(network_type, images_torch):
    print("network_type", network_type)
    distance_between_frames = 1
    if network_type in [NetworkType.DoubleRGB,NetworkType.DoubleRGBDepth]:
        # Ask the user for the distance between frames
        distance_between_frames = int(input("Enter the distance between frames: "))
        if distance_between_frames < 1:
            raise ValueError("Distance between frames must be greater than 0.")

    if network_type == NetworkType.DoubleRGB:
        doubled_images = []
        for i in range(len(images_torch)):
            # Handle the case where the current index minus distance is negative
            prev_index = max(i - distance_between_frames, 0)
            doubled_images.append(torch.cat((images_torch[prev_index], images_torch[i]), dim=0))
        images_torch = torch.stack(doubled_images)
        print(f"Double RGB images with distance {distance_between_frames} frames. New shape: {images_torch.shape}")

    if network_type == NetworkType.DoubleRGBDepth:
        doubled_images = []
        for i in range(len(images_torch)):
            # Handle the case where the current index minus distance is negative
            prev_index = max(i - distance_between_frames, 0)
            doubled_images.append(torch.cat((images_torch[prev_index], images_torch[i]), dim=0))
        images_torch = torch.stack(doubled_images)
        print(f"Double RGBDepth images with distance {distance_between_frames} frames. New shape: {images_torch.shape}")

    return images_torch

