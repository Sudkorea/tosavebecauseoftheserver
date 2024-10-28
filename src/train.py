from processors.tabular_processor import TabularProcessor
from models.lgbm_model import LGBMModel
import wandb
import pandas as pd

def main():
    # 데이터 로드
    train_data = pd.read_csv('../data/train.csv')
    
    # 데이터 전처리
    processor = TabularProcessor()
    processed_data = processor.fit(train_data).process(train_data)
    
    # 특징 선택
    features = ['area_m2', 'contract_year_month', 'contract_day', 
               'floor', 'latitude', 'longitude', 'age', 'cluster']
    X = processed_data[features]
    y = processed_data['deposit']
    
    # 모델 학습
    model = LGBMModel()
    model.train(X, y)
    
    # 모델 저장
    model.save('../models/lgbm_model.txt')

if __name__ == "__main__":
    main()
