from .base_processor import BaseProcessor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

class TabularProcessor(BaseProcessor):
    """테이블 데이터 전처리 클래스"""
    
    def __init__(self, n_clusters=25, random_state=42):
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        
    def fit(self, data):
        """위치 데이터를 기반으로 클러스터링 학습"""
        location_data = data[['latitude', 'longitude']].copy()
        location_scaled = self.scaler.fit_transform(location_data)
        self.kmeans.fit(location_scaled)
        return self
        
    def process(self, data):
        """데이터 전처리 및 클러스터 할당"""
        processed_data = data.copy()
        
        # 위치 데이터 스케일링 및 클러스터링
        location_data = data[['latitude', 'longitude']].copy()
        location_scaled = self.scaler.transform(location_data)
        clusters = self.kmeans.predict(location_scaled)
        
        # 클러스터 정보 추가
        processed_data['cluster'] = clusters
        
        return processed_data
