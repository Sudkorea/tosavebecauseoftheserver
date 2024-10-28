
# 코드 템플릿 (Code Templates)

## 클래스 템플릿

모든 클래스는 다음 형식을 따릅니다:

```python
from typing import Optional, List, Dict, Any

class ClassName:
    """클래스에 대한 설명
    
    Attributes:
        attribute1 (type): 설명
        attribute2 (type): 설명
    """
    
    def __init__(self, param1: type1, param2: Optional[type2] = None):
        """초기화 메서드
        
        Args:
            param1 (type1): 설명
            param2 (type2, optional): 설명. Defaults to None.
        """
        self.attribute1 = param1
        self.attribute2 = param2
    
    def method_name(self, param: type) -> return_type:
        """메서드 설명
        
        Args:
            param (type): 설명
            
        Returns:
            return_type: 반환값 설명
            
        Raises:
            ErrorType: 예외 발생 조건 설명
        """
        pass
```

## 프로세서 템플릿

모든 프로세서는 BaseProcessor를 상속받아 구현합니다:

```python
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseProcessor(ABC):
    """데이터 처리를 위한 기본 프로세서 클래스"""
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        """데이터 처리 메서드
        
        Args:
            data (Any): 처리할 데이터
            
        Returns:
            Any: 처리된 데이터
        """
        pass
```

## 모델 템플릿

모든 모델은 BaseModel을 상속받아 구현합니다:

```python
class BaseModel(ABC):
    """모델 구현을 위한 기본 클래스"""
    
    def __init__(self, input_dim: int, output_dim: int):
        """
        모델 초기화
        :param input_dim: 입력 차원
        :param output_dim: 출력 차원
        """
        super(BaseModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파 정의
        :param x: 입력 데이터
        :return: 모델 출력
        """
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x
        
    @abstractmethod
    def train(self, train_data: Any, valid_data: Optional[Any] = None) -> Dict[str, float]:
        """모델 학습 메서드
        
        Args:
            train_data (Any): 학습 데이터
            valid_data (Optional[Any]): 검증 데이터
            
        Returns:
            Dict[str, float]: 학습 결과 메트릭
        """
        pass
    
    @abstractmethod 
    def predict(self, data: Any) -> Any:
        """예측 메서드
        
        Args:
            data (Any): 예측할 데이터
            
        Returns:
            Any: 예측 결과
        """
        pass
```

## 설정 파일 템플릿 (config.yaml)

```yaml
model:
  name: "모델명"
  params:
    param1: value1
    param2: value2

data:
  train_path: "경로"
  valid_path: "경로"
  test_path: "경로"
  
training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  
logging:
  log_dir: "logs/"
  save_dir: "models/"
```

# 템플릿 사용 가이드라인

1. **문서화 규칙**:
   - 모든 클래스와 메서드는 docstring을 포함해야 합니다
   - Type hints를 사용하여 인자와 반환 타입을 명시합니다
   - 주요 로직에는 인라인 주석을 추가합니다

2. **명명 규칙**:
   - 클래스: PascalCase (예: TextProcessor)
   - 메서드/함수: snake_case (예: process_data)
   - 변수: snake_case (예: train_data)
   - 상수: UPPER_CASE (예: MAX_BATCH_SIZE)

3. **에러 처리**:
   - 예상 가능한 에러는 명시적으로 처리합니다
   - 커스텀 예외를 정의하여 사용합니다
   - 에러 메시지는 명확하고 구체적이어야 합니다

4. **로깅**:
   - 중요한 처리 단계마다 로그를 남깁니다
   - 에러와 경고는 적절한 로그 레벨을 사용합니다
   - 실험 결과와 메트릭을 체계적으로 기록합니다

이러한 템플릿과 가이드라인을 따름으로써 일관된 코드 스타일을 유지하고, 팀원 간의 협업을 원활하게 할 수 있습니다.
