from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any

class BaseProcessor(ABC):
    """추천 시스템을 위한 데이터 전처리 기본 클래스"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        초기화
        Args:
            config: 전처리 설정값을 담은 딕셔너리
        """
        self.config = config or {}
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, data: Dict[str, pd.DataFrame]) -> None:
        """전처리 파라미터 학습"""
        pass
    
    @abstractmethod
    def transform(self, data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """
        데이터를 모델 입력에 적합한 형태로 변환
        
        Returns:
            {
                'user_features': np.ndarray,    # (n_users, user_feature_dim)
                'item_features': np.ndarray,    # (n_items, item_feature_dim)
                'interaction_matrix': np.ndarray # (n_users, n_items)
            }
        """
        pass
