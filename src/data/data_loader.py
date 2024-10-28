import pandas as pd
import os
from typing import Dict, Optional

class DataLoader:
    """데이터 로딩을 위한 클래스"""
    
    def __init__(self, data_dir: str):
        """
        데이터 로더 초기화
        
        Args:
            data_dir: 데이터 디렉토리 경로
        """
        self.data_dir = data_dir
        self._validate_data_dir()
    
    def _validate_data_dir(self):
        """데이터 디렉토리 유효성 검사"""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"데이터 디렉토리를 찾을 수 없습니다: {self.data_dir}")
    
    def load_all(self) -> Dict[str, pd.DataFrame]:
        """
        모든 데이터 파일 로드
        
        Returns:
            Dict[str, pd.DataFrame]: 데이터프레임을 포함하는 딕셔너리
        """
        data = {
            'books': self.load_books(),
            'users': self.load_users(),
            'train_ratings': self.load_train_ratings(),
            'test_ratings': self.load_test_ratings()
        }
        return data
    
    def load_books(self) -> pd.DataFrame:
        """books.csv 로드"""
        path = os.path.join(self.data_dir, 'books.csv')
        return pd.read_csv(path)
    
    def load_users(self) -> pd.DataFrame:
        """users.csv 로드"""
        path = os.path.join(self.data_dir, 'users.csv')
        return pd.read_csv(path)
    
    def load_train_ratings(self) -> pd.DataFrame:
        """train_ratings.csv 로드"""
        path = os.path.join(self.data_dir, 'train_ratings.csv')
        return pd.read_csv(path)
    
    def load_test_ratings(self) -> pd.DataFrame:
        """test_ratings.csv 로드"""
        path = os.path.join(self.data_dir, 'test_ratings.csv')
        return pd.read_csv(path)
    
    def get_image_path(self, book_id: str, size: str = 'medium') -> str:
        """
        책 이미지 파일 경로 반환
        
        Args:
            book_id: 책 ID
            size: 이미지 크기 ('original' 또는 'medium')
            
        Returns:
            str: 이미지 파일 경로
        """
        folder = 'images'
        return os.path.join(self.data_dir, folder, f"{book_id}.jpg")
