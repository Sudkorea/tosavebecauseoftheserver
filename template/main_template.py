import argparse
from data_template import load_data, preprocess_data, create_dataloader
from model_template import CustomModel
from train_template import train_model, evaluate_model
from utils_template import set_seed
import torch
import torch.nn as nn
import torch.optim as optim

def main(args):
    """
    메인 함수
    :param args: 명령행 인자
    """
    set_seed(args.seed)

    # 데이터 로드 및 전처리
    data = load_data(args.data_path)
    data = preprocess_data(data)
    dataloader = create_dataloader(data, args.batch_size)

    # 모델 초기화
    model = CustomModel(input_dim=args.input_dim, output_dim=args.output_dim).to(args.device)

    # 손실 함수 및 옵티마이저 설정
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 모델 훈련 및 평가
    train_model(model, dataloader, criterion, optimizer, args.device)
    evaluate_model(model, dataloader, criterion, args.device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--input_dim', type=int, required=True)
    parser.add_argument('--output_dim', type=int, required=True)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)
