from abc import ABC, abstractmethod

class BaseModel(ABC):
    """모델 구현을 위한 기본 클래스"""
    
    @abstractmethod
    def train(self, X, y):
        """모델 학습"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """예측 수행"""
        pass
    
    @abstractmethod
    def save(self, path):
        """모델 저장"""
        pass
    
    @abstractmethod
    def load(self, path):
        """모델 로드"""
        pass
