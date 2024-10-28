import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import wandb
from models.ncf import NCF
from processors.simple_processor import SimpleProcessor
from data.data_loader import DataLoader as DataLoad

def train_epoch(model, train_loader, criterion, optimizer, device):
    """한 에폭 학습"""
    model.train()
    total_loss = 0
    
    with tqdm(train_loader, desc='Training') as t:
        for user_indices, item_indices, ratings in t:
            user_indices = user_indices.to(device)
            item_indices = item_indices.to(device)
            ratings = ratings.to(device)
            
            # Forward pass
            predictions = model(user_indices, item_indices)
            loss = criterion(predictions, ratings)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            t.set_postfix({'loss': f'{loss.item():.4f}'})
            
    return total_loss / len(train_loader)

def validate(model, valid_loader, criterion, device):
    """검증"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for user_indices, item_indices, ratings in valid_loader:
            user_indices = user_indices.to(device)
            item_indices = item_indices.to(device)
            ratings = ratings.to(device)
            
            predictions = model(user_indices, item_indices)
            loss = criterion(predictions, ratings)
            total_loss += loss.item()
            
    return total_loss / len(valid_loader)

def main(config):
    # 설정
    device = torch.device(config.device)
    
    # 데이터 로드 및 전처리
    data_loader = DataLoad(config.dataset.data_path)
    data = data_loader.load_all()
    
    processor = SimpleProcessor()
    processed_data = processor.fit_transform(data)
    
    # 데이터셋 생성
    user_indices = torch.LongTensor(processed_data['user_indices'])
    item_indices = torch.LongTensor(processed_data['item_indices'])
    ratings = torch.FloatTensor(processed_data['ratings'])
    
    # 데이터 로더 생성
    dataset = TensorDataset(user_indices, item_indices, ratings)
    train_size = int(len(dataset) * (1 - config.dataset.valid_ratio))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dataloader.batch_size,
        shuffle=config.dataloader.shuffle,
        num_workers=config.dataloader.num_workers
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.dataloader.batch_size,
        shuffle=False,
        num_workers=config.dataloader.num_workers
    )
    
    # 모델 초기화
    model = NCF(config.model_args.NCF, processed_data).to(device)
    
    # 손실 함수 및 옵티마이저 설정
    criterion = getattr(nn, config.loss)()
    optimizer = getattr(optim, config.optimizer.type)(
        model.parameters(),
        **config.optimizer.args
    )
    
    # 학습률 스케줄러 설정
    if config.lr_scheduler.use:
        scheduler = getattr(optim.lr_scheduler, config.lr_scheduler.type)(
            optimizer,
            **config.lr_scheduler.args
        )
    
    # 학습 시작
    best_loss = float('inf')
    early_stopping_counter = 0
    
    for epoch in range(config.train.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss = validate(model, valid_loader, criterion, device)
        
        if config.lr_scheduler.use:
            scheduler.step(valid_loss)
        
        # Wandb 로깅
        if wandb.run is not None:
            wandb.log({
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'lr': optimizer.param_groups[0]['lr']
            })
        
        # 모델 저장
        if valid_loss < best_loss:
            best_loss = valid_loss
            early_stopping_counter = 0
            save_path = Path(config.train.save_dir) / f'{config.model}_best.pth'
            torch.save(model.state_dict(), save_path)
        else:
            early_stopping_counter += 1
            
        if early_stopping_counter >= config.train.early_stopping:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break
            
    return model

if __name__ == "__main__":
    import yaml
    from omegaconf import OmegaConf
    
    # 설정 파일 로드
    with open('configs/ncf_config.yaml') as f:
        config = OmegaConf.create(yaml.safe_load(f))
    
    # Wandb 초기화
    if config.get('wandb', False):
        wandb.init(project=config.wandb_project, config=config)
    
    # 학습 실행
    model = main(config)
