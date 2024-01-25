"""
Copyright (c) Bas Blokzijl, Leiden University 2023-2024.

Module for debugging data preparation.
"""	

import matplotlib.pyplot as plt

def debug_plot(image_dataset, vehicle_data, labels):
    """
    @brief Displays a user-specified number of images along with their associated data and labels.
    @param image_dataset: The TensorFlow dataset object containing the images.
    @param vehicle_data: The NumPy array containing the LastSteering and Speed data.
    @param labels: The NumPy array containing the Throttle and Steering labels.
    """
    num_images = len(vehicle_data)
    print(f"(1 - {num_images})")
    num_to_display = int(input("Enter the number of images you want to see: "))
    
    # Use the 'take' method to get the first 'num_to_display' images.
    images_to_display = image_dataset.take(num_to_display)
    
    for idx, image in enumerate(images_to_display):
        image_np = image.numpy()
        if image_np.shape[0] == 1:


            image_np = image_np.squeeze(0)
        current_vehicle_data = vehicle_data[idx]
        current_labels = labels[idx]
        
        plt.figure()
        plt.imshow(image_np)
        plt.title(f'LastSteering: {current_vehicle_data[0]}, Speed: {current_vehicle_data[1]}, Throttle: {current_labels[0]}, Steering: {current_labels[1]}')
        plt.show()


def debug_images(images, vehicle_data_batch, labels_batch, frame_numbers):
    """
    Debugs all images in a batch along with their associated data and labels.
    
    :param images: Batch of images.
    :param vehicle_data_batch: Batch of vehicle data.
    :param labels_batch: Batch of labels.
    :param frame_numbers: Numbers of the frames being processed.
    """
    for i in range(len(images)):
        image = images[i].cpu().numpy().transpose(1, 2, 0)
        vehicle_data = vehicle_data_batch[i]
        label = labels_batch[i]
        frame_number = frame_numbers[i]

        plt.figure()
        plt.imshow(image)
        plt.title(f'Frame: {frame_number}, LastSteering: {vehicle_data[0]}, Speed: {vehicle_data[1]}, Throttle: {label[0]}, Steering: {label[1]}')
        plt.show()

def debug_depth_data(image_dataset, vehicle_data, labels, indices):
    """
    @brief Displays specified images along with their associated data and labels.
    @param image_dataset: The TensorFlow dataset object containing the concatenated images (RGB + Depth).
    @param vehicle_data: The NumPy array containing the LastSteering and Speed data.
    @param labels: The NumPy array containing the Throttle and Steering labels.
    @param indices: A list of indices specifying which images to display.
    """
    for idx in indices:
        if idx < 0 or idx >= len(image_dataset):
            print(f"Index {idx} is out of range. Skipping...")
            continue
        
        image = image_dataset.skip(idx).take(1)
        image = list(image)[0]
        
        # Split the concatenated image tensor into RGB and Depth images.
        rgb_image = image[..., :3].numpy()
        depth_image = image[..., 3:].numpy()
        
        current_vehicle_data = vehicle_data[idx]
        current_labels = labels[idx]
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(rgb_image)
        plt.title(f'RGB Image\nLastSteering: {current_vehicle_data[0]}, Speed: {current_vehicle_data[1]}, Throttle: {current_labels[0]}, Steering: {current_labels[1]}')
        
        plt.subplot(1, 2, 2)
        plt.imshow(depth_image.squeeze(-1), cmap='gray') 
        plt.title(f'Depth Image\nLastSteering: {current_vehicle_data[0]}, Speed: {current_vehicle_data[1]}, Throttle: {current_labels[0]}, Steering: {current_labels[1]}')
        
        plt.tight_layout()
        plt.show()

def debug_pureDepth_data(image_dataset, vehicle_data, labels, indices):
    """
    @brief Displays specified depth images along with their associated data and labels.
    @param image_dataset: The TensorFlow dataset object containing the depth images.
    @param vehicle_data: The NumPy array containing the LastSteering and Speed data.
    @param labels: The NumPy array containing the Throttle and Steering labels.
    @param indices: A list of indices specifying which images to display.
    """
    for idx in indices:
        if idx < 0 or idx >= len(image_dataset):
            print(f"Index {idx} is out of range. Skipping...")
            continue

        # Use the 'skip' and 'take' methods to get the specified image.
        depth_image = image_dataset.skip(idx).take(1)
        depth_image = list(depth_image)[0]  # Convert tf.data.Dataset to a list to access the image tensor

        # Assuming depth image is single channel, squeeze and use grayscale colormap
        depth_image_np = depth_image.numpy().squeeze(-1)

        current_vehicle_data = vehicle_data[idx]
        current_labels = labels[idx]

        # Plot the Depth image
        plt.figure(figsize=(5, 5))
        plt.imshow(depth_image_np, cmap='gray')
        plt.title(f'Depth Image\nLastSteering: {current_vehicle_data[0]}, Speed: {current_vehicle_data[1]}, Throttle: {current_labels[0]}, Steering: {current_labels[1]}')
        plt.tight_layout()
        plt.show()