import argparse
import ast
from omegaconf import OmegaConf
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import wandb

from data.data_loader import DataLoader
from processors.simple_processor import SimpleProcessor
from models.ncf import NCF

def main(args):
    # 설정 로드
    if args.config:
        config = OmegaConf.load(args.config)
        # 명령행 인자로 config 덮어쓰기
        for key, value in vars(args).items():
            if value is not None:
                config[key] = value
    else:
        config = OmegaConf.create(vars(args))

    # 데이터 로드
    print("Loading data...")
    data_loader = DataLoader(config.dataset.data_path)
    data = data_loader.load_all()
    
    # 전처리
    print("Preprocessing data...")
    processor = SimpleProcessor()
    processed_data = processor.fit_transform(data)
    
    # 모델 초기화
    print(f"Initializing {config.model} model...")
    model = NCF(config.model_args[config.model], processed_data).to(config.device)
    
    # 체크포인트에서 모델 로드
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint))
    
    # 학습 모드
    if not args.predict:
        print("Starting training...")
        criterion = getattr(nn, config.loss)()
        optimizer = getattr(optim, config.optimizer.type)(
            model.parameters(),
            **config.optimizer.args
        )
        
        scheduler = None
        if config.lr_scheduler.use:
            scheduler = getattr(optim.lr_scheduler, config.lr_scheduler.type)(
                optimizer,
                **config.lr_scheduler.args
            )
        
        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=config.device
        )
        
        trainer.train(
            train_loader=processed_data['train_loader'],
            valid_loader=processed_data['valid_loader'],
            config=config
        )
    
    # 추론 모드
    print("Generating predictions...")
    predictions = trainer.predict(processed_data['test_loader'])
    
    # 결과 저장
    print("Saving predictions...")
    submission = pd.read_csv(Path(config.dataset.data_path) / 'sample_submission.csv')
    submission['rating'] = predictions
    submission.to_csv(Path(config.train.save_dir) / f'{config.model}_submission.csv', 
                     index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Book Rating Prediction')
    
    # 필수 인자
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--model', type=str, choices=['NCF'], help='Model to use')
    parser.add_argument('--predict', type=ast.literal_eval, help='Prediction mode')
    
    # 선택 인자
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    if args.predict and not args.checkpoint:
        parser.error("--predict requires --checkpoint")
    
    main(args)
