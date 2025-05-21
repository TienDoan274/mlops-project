import mlflow
import mlflow.pytorch
import torch
from torchvision.models import resnet18
import numpy as np
import os

def load_data(data_dir):
    # Tái sử dụng hàm load_data từ train.py
    images = []
    labels = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.npy'):
            img = np.load(os.path.join(data_dir, filename))
            images.append(img)
            label = 1 if 'dog' in filename.lower() else 0
            labels.append(label)
    return torch.tensor(images), torch.tensor(labels)

def evaluate_model(model_path, data_dir):
    model = resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    X, y = load_data(data_dir)
    with torch.no_grad():
        outputs = model(X.to(device))
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y.to(device)).float().mean().item()
    
    with mlflow.start_run():
        mlflow.log_metric("eval_accuracy", accuracy)
        if accuracy > 0.85:
            mlflow.pytorch.log_model(model, "model")
            mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", "DogCatModel")

if __name__ == "__main__":
    import sys
    evaluate_model(sys.argv[1], sys.argv[2])