import torch
from tqdm import tqdm
from pathlib import Path
import wandb
import numpy as np

class Trainer:
    """모델 학습을 담당하는 클래스"""
    
    def __init__(self, model, criterion, optimizer, scheduler=None, device='cuda'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
    def train_epoch(self, train_loader):
        """한 에폭 학습"""
        self.model.train()
        total_loss = 0
        
        with tqdm(train_loader, desc='Training') as t:
            for batch in t:
                # 데이터를 디바이스로 이동
                batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                
                # 순전파
                predictions = self.model(batch['input'])
                loss = self.criterion(predictions, batch['label'])
                
                # 역전파
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                t.set_postfix({'loss': f'{loss.item():.4f}'})
                
        return total_loss / len(train_loader)
    
    def validate(self, valid_loader):
        """검증"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in valid_loader:
                batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                predictions = self.model(batch['input'])
                loss = self.criterion(predictions, batch['label'])
                total_loss += loss.item()
                
        return total_loss / len(valid_loader)
    
    def train(self, train_loader, valid_loader, config):
        """전체 학습 과정"""
        best_loss = float('inf')
        early_stopping_counter = 0
        
        for epoch in range(config.train.epochs):
            # 학습
            train_loss = self.train_epoch(train_loader)
            valid_loss = self.validate(valid_loader)
            
            # 스케줄러 스텝
            if self.scheduler is not None:
                if config.lr_scheduler.type == 'ReduceLROnPlateau':
                    self.scheduler.step(valid_loss)
                else:
                    self.scheduler.step()
            
            # Wandb 로깅
            if wandb.run is not None:
                wandb.log({
                    'train_loss': train_loss,
                    'valid_loss': valid_loss,
                    'lr': self.optimizer.param_groups[0]['lr'],
                    'epoch': epoch + 1
                })
            
            # 모델 저장
            if valid_loss < best_loss:
                best_loss = valid_loss
                early_stopping_counter = 0
                save_path = Path(config.train.save_dir) / f'{config.model}_best.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'best_loss': best_loss,
                }, save_path)
            else:
                early_stopping_counter += 1
                
            print(f'Epoch {epoch+1}/{config.train.epochs} - '
                  f'Train Loss: {train_loss:.4f} - '
                  f'Valid Loss: {valid_loss:.4f}')
            
            if early_stopping_counter >= config.train.early_stopping:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break
                
        return best_loss
    
    def predict(self, test_loader):
        """추론"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                batch_predictions = self.model(batch['input'])
                predictions.extend(batch_predictions.cpu().numpy())
                
        return np.array(predictions)
