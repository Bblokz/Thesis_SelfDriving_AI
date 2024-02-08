from PIL import Image
import os

def make_more_blue(image_path, output_path, blue_factor):
    # Open an image file
    with Image.open(image_path) as img:
        # Convert the image to RGB if it's not
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Load the pixel data
        pixels = img.load()

        # Adjust the blue channel
        for i in range(img.width):
            for j in range(img.height):
                r, g, b = pixels[i, j]
                pixels[i, j] = (r, g, min(255, int(b * blue_factor)))

        # Save the modified image
        img.save(output_path)

def main():
    input_directory = '.'
    output_directory = '.'

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Blue factor (change this to adjust the level of blue)
    blue_factor = 3 

    # Process each image in the directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".png"):
            input_path = os.path.join(input_directory, filename)
            output_path = os.path.join(output_directory, filename)
            make_more_blue(input_path, output_path, blue_factor)

if __name__ == '__main__':
    main()
