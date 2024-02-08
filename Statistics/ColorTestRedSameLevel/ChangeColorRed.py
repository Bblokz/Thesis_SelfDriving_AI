from PIL import Image
import os

def enhance_red_channel(image_path, output_path, red_factor):
    # Open an image file
    with Image.open(image_path) as img:
        # Convert the image to RGB if it's not
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Load the pixel data
        pixels = img.load()

        # Adjust the red channel
        for i in range(img.width):
            for j in range(img.height):
                r, g, b = pixels[i, j]
                pixels[i, j] = (min(255, int(r * red_factor)), g, b)

        # Save the modified image
        img.save(output_path)

def main():
    # Directory containing your images
    input_directory = '.'
    output_directory = './output'

    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Red factor (change this to adjust the level of red)
    red_factor = 3  # Example factor: greater than 1 increases the red component

    # Process each image in the directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".png"):
            input_path = os.path.join(input_directory, filename)
            output_path = os.path.join(output_directory, filename)
            enhance_red_channel(input_path, output_path, red_factor)

if __name__ == '__main__':
    main()
