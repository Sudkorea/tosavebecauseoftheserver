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

'''
아래는 모든 데이터셋 정보입니다.

books 데이터셋 정보:
Shape: (149570, 10)

Columns:
['isbn', 'book_title', 'book_author', 'year_of_publication', 'publisher', 'img_url', 'language', 'category', 'summary', 'img_path']

Sample data:
         isbn            book_title           book_author  year_of_publication              publisher                                            img_url language       category                                            summary                           img_path
0  0002005018          Clara Callan  Richard Bruce Wright               2001.0  HarperFlamingo Canada  http://images.amazon.com/images/P/0002005018.0...       en  ['Actresses']  In a small town in Canada, Clara Callan reluct...  images/0002005018.01.THUMBZZZ.jpg
1  0060973129  Decision in Normandy          Carlo D'Este               1991.0        HarperPerennial  http://images.amazon.com/images/P/0060973129.0...       en  ['1940-1949']  Here, for the first time in paperback, is an o...  images/0060973129.01.THUMBZZZ.jpg
--------------------------------------------------------------------------------

users 데이터셋 정보:
Shape: (68092, 3)

Columns:
['user_id', 'location', 'age']

Sample data:
   user_id                  location   age
0        8  timmins, ontario, canada   NaN
1    11400   ottawa, ontario, canada  49.0
--------------------------------------------------------------------------------

train_ratings 데이터셋 정보:
Shape: (306795, 3)

Columns:
['user_id', 'isbn', 'rating']

Sample data:
   user_id        isbn  rating
0        8  0002005018       4
1    67544  0002005018       7
--------------------------------------------------------------------------------

test_ratings 데이터셋 정보:
Shape: (76699, 3)

Columns:
['user_id', 'isbn', 'rating']

Sample data:
   user_id        isbn  rating
0    11676  0002005018       0
1   116866  0002005018       0
--------------------------------------------------------------------------------

샘플 이미지 경로: ../../data/images/0002005018.01.THUMBZZZ.jpg
'''