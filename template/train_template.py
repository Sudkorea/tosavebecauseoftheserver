import torch
import torch.optim as optim
import torch.nn as nn

def train_model(model, dataloader, criterion, optimizer, device):
    """
    모델 훈련
    :param model: 학습할 모델
    :param dataloader: 학습 데이터 로더
    :param criterion: 손실 함수
    :param optimizer: 옵티마이저
    :param device: 장치 (CPU/GPU)
    """
    model.train()
    for data, labels in dataloader:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def evaluate_model(model, dataloader, criterion, device):
    """
    모델 평가
    :param model: 평가할 모델
    :param dataloader: 평가 데이터 로더
    :param criterion: 손실 함수
    :param device: 장치 (CPU/GPU)
    :return: 평균 손실
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)
