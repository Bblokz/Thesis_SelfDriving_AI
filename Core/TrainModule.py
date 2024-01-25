"""
Copyright (c) Bas Blokzijl, Leiden University 2023-2024.

Module for training the models.
"""	

import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import tensorflow as tf
import torch
import numpy as np
import os

# Module imports.
from Core.DataManipulationModule import prepare_input_images
from Core.Constants import MODEL_PATH, MODEL_PATH_DEPTH, MODEL_PATH_PURE_DEPTH, MODEL_PATH_DOUBLE_RGB, MODEL_PATH_DOUBLE_RGB_DEPTH,  NetworkType

def train_model(image_dataset, vehicle_data, labels, model, device, network_type):
    # Determine the correct model path based on network type.
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
    else:
        raise ValueError(f'Invalid network type: {network_type}')
    
    # Check for saved model parameters and load them if available.
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded saved model parameters from {model_path}")
    else:
        print("No saved model parameters found. Training from scratch.")

    model.to(device)
    
    # Convert TensorFlow dataset to numpy array.
    images_np = np.array([image.numpy() for image in image_dataset])
    # normalise the image values between 0 and 1.
    images_torch = torch.tensor(images_np).permute(0, 3, 1, 2).float() / 255.0
    
    images_torch = prepare_input_images(network_type, images_torch)
        
    vehicle_data_torch = torch.tensor(vehicle_data).float()
    labels_torch = torch.tensor(labels).float()
    
    # Create a TensorDataset from the tensorsc
    dataset = TensorDataset(images_torch, vehicle_data_torch, labels_torch)
    data_loader = DataLoader(dataset, batch_size=48, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, eps=1e-08, weight_decay=0.0001)
    print(f"Weight decay: {optimizer.param_groups[0]['weight_decay']}")
    # MSE
    loss_fn = torch.nn.MSELoss()
    
    num_epochs = 500
    for epoch in range(num_epochs):
        for batch_idx, (images, vehicle_data_batch, labels_batch) in enumerate(data_loader):
            images, vehicle_data_batch, labels_batch = images.to(device), vehicle_data_batch.to(device), labels_batch.to(device)
            # reset the gradients.
            optimizer.zero_grad()
            
            
            # Forward pass: compute the model predictions
            throttle_pred, steering_pred = model(images, vehicle_data_batch)

            # Remove the last dimension as warning is given.
            throttle_pred = throttle_pred.squeeze(-1)  
            steering_pred = steering_pred.squeeze(-1) 
            
            # Compute the loss.
            loss = loss_fn(throttle_pred, labels_batch[:, 0]) + loss_fn(steering_pred, labels_batch[:, 1])

            # Compute the gradients.
            loss.backward()
        
            # Update the model parameters.
            optimizer.step()

        
        # Print the loss for this epoch.
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
    
    # Save the trained model parameters.
    torch.save(model.state_dict(), model_path)