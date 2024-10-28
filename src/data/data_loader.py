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
            'train': self.load_train(),
            'test': self.load_test(),
            'interest': self.load_interest(),
            'park': self.load_park(),
            'school': self.load_school(),
            'subway': self.load_subway()
        }
        return data
    
    def load_train(self) -> pd.DataFrame:
        """train.csv 로드"""
        path = os.path.join(self.data_dir, 'train.csv')
        return pd.read_csv(path)
    
    def load_test(self) -> pd.DataFrame:
        """test.csv 로드"""
        path = os.path.join(self.data_dir, 'test.csv')
        return pd.read_csv(path)
    
    def load_interest(self) -> pd.DataFrame:
        """interestRate.csv 로드"""
        path = os.path.join(self.data_dir, 'interestRate.csv')
        return pd.read_csv(path)
    
    def load_park(self) -> pd.DataFrame:
        """parkInfo.csv 로드"""
        path = os.path.join(self.data_dir, 'parkInfo.csv')
        return pd.read_csv(path)
    
    def load_school(self) -> pd.DataFrame:
        """schoolinfo.csv 로드"""
        path = os.path.join(self.data_dir, 'schoolinfo.csv')
        return pd.read_csv(path)
    
    def load_subway(self) -> pd.DataFrame:
        """subwayInfo.csv 로드"""
        path = os.path.join(self.data_dir, 'subwayInfo.csv')
        return pd.read_csv(path)
    
    def load_submission(self) -> Optional[pd.DataFrame]:
        """sample_submission.csv 로드"""
        path = os.path.join(self.data_dir, 'sample_submission.csv')
        if os.path.exists(path):
            return pd.read_csv(path)
        return None
