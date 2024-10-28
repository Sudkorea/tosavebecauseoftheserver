import pandas as pd
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, data, labels=None):
        """
        데이터셋 초기화
        :param data: 입력 데이터
        :param labels: 레이블 데이터 (옵션)
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        """
        데이터셋의 크기 반환
        :return: 데이터셋의 크기
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        인덱스에 해당하는 데이터 반환
        :param idx: 인덱스
        :return: 데이터 및 레이블 (레이블이 있는 경우)
        """
        item = self.data[idx]
        if self.labels is not None:
            label = self.labels[idx]
            return item, label
        return item

def load_data(file_path):
    """
    CSV 파일로부터 데이터 로드
    :param file_path: 파일 경로
    :return: 데이터프레임
    """
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """
    데이터 전처리
    :param data: 원본 데이터
    :return: 전처리된 데이터
    """
    # 예: 결측치 처리, 라벨 인코딩 등
    return data

def create_dataloader(data, batch_size, shuffle=True, num_workers=0):
    """
    DataLoader 생성
    :param data: 데이터셋
    :param batch_size: 배치 크기
    :param shuffle: 셔플 여부
    :param num_workers: 워커 수
    :return: DataLoader 객체
    """
    dataset = CustomDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader
