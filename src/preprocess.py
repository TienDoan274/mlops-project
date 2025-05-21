import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

def preprocess_data(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.png')):
            img = Image.open(os.path.join(input_dir, filename)).convert('RGB')
            img_tensor = transform(img)
            np.save(os.path.join(output_dir, f"{filename.split('.')[0]}.npy"), img_tensor.numpy())

if __name__ == "__main__":
    import sys
    preprocess_data(sys.argv[1], sys.argv[2])