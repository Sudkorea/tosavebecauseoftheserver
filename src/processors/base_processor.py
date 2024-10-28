from abc import ABC, abstractmethod
import pandas as pd

class BaseProcessor(ABC):
    """데이터 처리를 위한 기본 클래스"""
    
    @abstractmethod
    def process(self, data):
        """데이터 처리 메소드"""
        pass
    
    @abstractmethod
    def fit(self, data):
        """데이터 학습 메소드"""
        pass
