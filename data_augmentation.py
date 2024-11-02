import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Define the data augmentation transformations
data_transforms = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(224, scale=(0.85, 1.15)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15),
    transforms.ToTensor()
])

# Function to save the augmented images
def save_image(tensor, path):
    image = tensor.clone().detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    image.save(path)

# Directory paths
input_dir = "picture"
output_dir = "generater_pic"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# List all files in the input directory
dirs = os.listdir(input_dir)
print(len(dirs))

# Process each file
for filename in dirs:
    img_path = os.path.join(input_dir, filename)
    img = Image.open(img_path).convert("RGB")
    
    prefix = filename.split('.')[0]
    print(prefix)
    
    counter = 0
    for _ in range(101):  # Generate 101 augmented images
        augmented_img = data_transforms(img)
        save_path = os.path.join(output_dir, f"{prefix}_{counter}.jpg")
        save_image(augmented_img, save_path)
        counter += 1