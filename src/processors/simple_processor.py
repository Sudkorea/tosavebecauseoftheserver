from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any
from .base_processor import BaseProcessor
from torch.utils.data import DataLoader, TensorDataset
import torch

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
        
        # field_dims 계산 (NCF 모델에서 필요)
        self.field_dims = [len(self.user_map), len(self.book_map)]
        
        # 수치형 데이터의 통계량 계산
        self.age_mean = users['age'].mean()
        self.age_std = users['age'].std()
        self.year_mean = books['year_of_publication'].mean()
        
        # 이상치 기준 설정 (나이)
        self.age_min = 0
        self.age_max = 100
        
        self.is_fitted = True
    
    def transform(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        실제 전처리 수행
        """
        if not self.is_fitted:
            raise RuntimeError("transform 전에 fit을 먼저 호출해야 합니다.")
            
        users = data['users'].copy()
        books = data['books'].copy()
        ratings = data.get('train_ratings', data.get('test_ratings')).copy()
        
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
        
        # 데이터 로더 생성을 위한 정보 추가
        processed_data = {
            'user_features': user_features,
            'item_features': item_features,
            'interaction_matrix': interaction_matrix,
            'field_dims': self.field_dims,  # NCF 모델을 위한 field_dims 추가
            'user_map': self.user_map,
            'book_map': self.book_map
        }
        
        # 데이터 로더 생성
        if 'train_ratings' in data:
            train_data = self._prepare_loader_data(data['train_ratings'])
            # 학습/검증 데이터 분할 (예: 8:2)
            n_samples = len(train_data['input'])
            n_train = int(0.8 * n_samples)
            indices = np.random.permutation(n_samples)
            
            # 학습 데이터 로더
            train_dataset = TensorDataset(
                torch.tensor(train_data['input'][indices[:n_train]], dtype=torch.long),
                torch.tensor(train_data['label'][indices[:n_train]], dtype=torch.float)
            )
            processed_data['train_loader'] = DataLoader(
                train_dataset, 
                batch_size=128,  # 이 값은 config에서 받아올 수 있습니다
                shuffle=True
            )
            
            # 검증 데이터 로더
            valid_dataset = TensorDataset(
                torch.tensor(train_data['input'][indices[n_train:]], dtype=torch.long),
                torch.tensor(train_data['label'][indices[n_train:]], dtype=torch.float)
            )
            processed_data['valid_loader'] = DataLoader(
                valid_dataset,
                batch_size=256,  # 이 값은 config에서 받아올 수 있습니다
                shuffle=False
            )
        
        if 'test_ratings' in data:
            test_data = self._prepare_loader_data(data['test_ratings'])
            test_dataset = TensorDataset(
                torch.tensor(test_data['input'], dtype=torch.long),
                torch.tensor(np.zeros(len(test_data['input'])), dtype=torch.float)  # 더미 레이블
            )
            processed_data['test_loader'] = DataLoader(
                test_dataset,
                batch_size=256,  # 이 값은 config에서 받아올 수 있습니다
                shuffle=False
            )
        
        return processed_data
    
    def _prepare_loader_data(self, ratings: pd.DataFrame) -> Dict[str, np.ndarray]:
        """데이터 로더를 위한 데이터 준비"""
        user_indices = np.array([self.user_map[uid] for uid in ratings['user_id']])
        item_indices = np.array([self.book_map[isbn] for isbn in ratings['isbn']])
        
        input_data = np.column_stack([user_indices, item_indices])
        labels = ratings['rating'].values if 'rating' in ratings else None
        
        return {
            'input': input_data,
            'label': labels
        }
