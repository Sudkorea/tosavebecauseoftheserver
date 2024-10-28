from .base_model import BaseModel
import lightgbm as lgb
import wandb
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class LGBMModel(BaseModel):
    """LightGBM 모델 클래스"""
    
    def __init__(self, params=None):
        self.params = params or {
            'objective': 'regression',
            'metric': ['mae', 'rmse'],
            'boosting_type': 'gbdt',
            'num_leaves': 1000,
            'learning_rate': 0.05,
            'n_estimators': 1000,
            'verbose': -1
        }
        self.model = None
        
    def train(self, X, y, valid_sets=None):
        """모델 학습"""
        train_data = lgb.Dataset(X, label=y)
        
        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=valid_sets,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )
        
        return self
    
    def predict(self, X):
        """예측 수행"""
        return self.model.predict(X)
    
    def save(self, path):
        """모델 저장"""
        self.model.save_model(path)
        
    def load(self, path):
        """모델 로드"""
        self.model = lgb.Booster(model_file=path)
