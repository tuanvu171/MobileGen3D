from PIL import Image
import os

scene_name = "woman-512" # modify if necessary
image_size = 512

# Define source and destination directories
source_directory = "./renders/"+scene_name+"/images"
destination_directory = "../dataset/nerf_data/"+scene_name+"/images"

# Create the destination directory if it doesn't exist
os.makedirs(destination_directory, exist_ok=True)

# Iterate through the source directory
for filename in os.listdir(source_directory):
    source_path = os.path.join(source_directory, filename)
    
    # Open the image using PIL
    img = Image.open(source_path)

    # Calculate the dimensions for resizing and cropping
    width, height = img.size
    size = (image_size, image_size)
    
    if width > height:
        left = (width - height) / 2
        top = 0
        right = (width + height) / 2
        bottom = height
    else:
        left = 0
        top = (height - width) / 4  # You can adjust this value to prefer cutting top over bottom
        right = width
        bottom = height - (height - width) / 4
    
    # Crop and resize the image
    img = img.crop((left, top, right, bottom))
    img = img.resize(size, Image.LANCZOS)

    # Save the processed image to the destination directory
    destination_path = os.path.join(destination_directory, filename)
    img.save(destination_path)
