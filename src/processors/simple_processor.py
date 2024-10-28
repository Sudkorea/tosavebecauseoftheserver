from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any
from .base_processor import BaseProcessor

class SimpleProcessor(BaseProcessor):
    """기본적인 전처리를 수행하는 클래스"""
    
    def fit(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        전처리 파라미터 학습 (예: 평균값, ID 매핑 등)
        """
        users = data['users']
        books = data['books']
        
        # 사용자/아이템 ID 매핑
        self.user_map = {uid: idx for idx, uid in enumerate(users['user_id'].unique())}
        self.book_map = {isbn: idx for idx, isbn in enumerate(books['isbn'].unique())}
        
        # 수치형 데이터의 통계량 계산
        self.age_mean = users['age'].mean()
        self.age_std = users['age'].std()
        self.year_mean = books['year_of_publication'].mean()
        
        # 이상치 기준 설정 (나이)
        self.age_min = 0
        self.age_max = 100
        
        self.is_fitted = True
    
    def transform(self, data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """
        실제 전처리 수행
        """
        if not self.is_fitted:
            raise RuntimeError("transform 전에 fit을 먼저 호출해야 합니다.")
            
        users = data['users'].copy()
        books = data['books'].copy()
        ratings = data['train_ratings'].copy()
        
        # 1. 사용자 특징 전처리
        users['age'] = users['age'].fillna(self.age_mean)
        users['age'] = users['age'].clip(self.age_min, self.age_max)  # 이상치 처리
        users['age'] = (users['age'] - self.age_mean) / self.age_std  # 표준화
        
        # 2. 책 특징 전처리
        books['year_of_publication'] = books['year_of_publication'].fillna(self.year_mean)
        books['year_of_publication'] = books['year_of_publication'].astype(float)
        
        # 출판연도 이상치 처리 (1900년 이전은 1900년으로, 현재 이후는 현재년도로)
        current_year = pd.Timestamp.now().year
        books['year_of_publication'] = books['year_of_publication'].clip(1900, current_year)
        
        # 3. 상호작용 행렬 생성
        n_users = len(self.user_map)
        n_items = len(self.book_map)
        interaction_matrix = np.zeros((n_users, n_items))
        
        for _, row in ratings.iterrows():
            if row['user_id'] in self.user_map and row['isbn'] in self.book_map:
                u_idx = self.user_map[row['user_id']]
                i_idx = self.book_map[row['isbn']]
                interaction_matrix[u_idx, i_idx] = row['rating']
        
        # 최종 특징 생성
        user_features = np.column_stack([
            users['age'].values,
            # 여기에 추가 사용자 특징을 더할 수 있음
        ])
        
        item_features = np.column_stack([
            (books['year_of_publication'] - 1900) / (current_year - 1900),  # 출판연도 정규화
            # 여기에 추가 책 특징을 더할 수 있음
        ])
        
        return {
            'user_features': user_features,
            'item_features': item_features,
            'interaction_matrix': interaction_matrix
        }
    
    def _validate_ratings(self, ratings: np.ndarray) -> np.ndarray:
        """평점 데이터 검증 및 정규화"""
        return np.clip(ratings, 0, 10) / 10.0  # 0-10 범위의 평점을 0-1로 정규화
