import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
import os

def build_model():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    # Đóng băng các tầng convolutional
    for param in model.parameters():
        param.requires_grad = False
    # Thay đổi tầng fully connected cho 2 lớp (chó và mèo)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

def load_data(data_dir):
    images = []
    labels = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.npy'):
            img = np.load(os.path.join(data_dir, filename))
            images.append(img)
            label = 1 if 'dog' in filename.lower() else 0  # dog: 1, cat: 0
            labels.append(label)
    return torch.tensor(images), torch.tensor(labels)

if __name__ == "__main__":
    import sys
    data_dir, model_path = sys.argv[1], sys.argv[2]
    
    # Tải dữ liệu
    X, y = load_data(data_dir)
    
    # Thiết lập mô hình
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
    # MLflow tracking
    mlflow.set_experiment("dog_cat_classification")
    with mlflow.start_run():
        model.train()
        for epoch in range(5):
            optimizer.zero_grad()
            outputs = model(X.to(device))
            loss = criterion(outputs, y.to(device))
            loss.backward()
            optimizer.step()
            mlflow.log_metric("loss", loss.item(), step=epoch)
        
        # Đánh giá
        model.eval()
        with torch.no_grad():
            outputs = model(X.to(device))
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y.to(device)).float().mean().item()
            mlflow.log_metric("accuracy", accuracy)
        
        # Lưu mô hình
        torch.save(model.state_dict(), model_path)
        mlflow.pytorch.log_model(model, "model")