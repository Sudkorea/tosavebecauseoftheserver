```markdown
# 프로젝트 개요 (Project Overview)

본 프로젝트는 사용자의 책 평점 데이터를 활용하여 사용자가 어떤 책을 더 선호할지 예측하는 것을 목표로 합니다. 이를 위해 다양한 메타데이터와 사용자-아이템 상호작용 데이터를 통합하여 추천 시스템 모델을 구축합니다.

주요 데이터는 다음과 같습니다:

- **books.csv**: 149,570개의 책에 대한 메타데이터.
- **users.csv**: 68,092명의 사용자에 대한 메타데이터.
- **train_ratings.csv**: 59,803명의 사용자가 129,777개의 책에 대해 남긴 306,795건의 평점 데이터.
- **test_ratings.csv**: 예측 대상인 사용자와 책 목록으로, 평점은 0으로 표시되어 있습니다.
- **이미지 데이터**: 각 책의 표지 이미지 파일.

# 모듈화 구조 (Modular Structure)

프로젝트는 다음과 같은 모듈화된 구조로 구성됩니다:

## 핵심 모듈 (Core Modules)

- **데이터 처리 (processors/)**
  - `base_processor.py`: 데이터 처리를 위한 기본 클래스
  - `text_processor.py`: 텍스트 데이터 전처리 클래스
  - `image_processor.py`: 이미지 데이터 전처리 클래스
  - `tabular_processor.py`: 테이블 데이터 전처리 클래스

- **특징 공학 (features/)**
  - `base_feature.py`: 특징 추출을 위한 기본 클래스
  - `text_feature.py`: 텍스트 특징 추출 클래스
  - `image_feature.py`: 이미지 특징 추출 클래스
  - `interaction_feature.py`: 상호작용 특징 추출 클래스

- **모델 (models/)**
  - `base_model.py`: 모델 구현을 위한 기본 클래스
  - `collaborative_model.py`: 협업 필터링 모델
  - `content_model.py`: 콘텐츠 기반 필터링 모델
  - `hybrid_model.py`: 하이브리드 추천 모델

## 실행 스크립트 (Main Scripts)

- **main.py**: 모델 학습 및 추론을 위한 통합 스크립트
  - 명령행 인자를 통한 모드 설정 (학습/추론)
  - 설정 파일(config.yaml)과 명령행 인자를 통한 하이퍼파라미터 관리
  - 데이터 로드 및 전처리
  - 모델 학습/추론 실행
  
사용 예시:
```bash
# 학습 모드
python main.py --config configs/ncf_config.yaml --model NCF --predict False

# 추론 모드
python main.py --config configs/ncf_config.yaml --model NCF --predict True --checkpoint models/NCF_best.pth
```

## 유틸리티 (Utils)

- **utils/**
  - `config.py`: 설정 관리
  - `metrics.py`: 평가 지표 계산
  - `logger.py`: 로깅 유틸리티
  - `data_loader.py`: 데이터 로딩 유틸리티

# 파일 구조 (File Structure)

```
project/
├── src/
│   ├── processors/
│   ├── features/
│   ├── models/
│   ├── utils/
│   ├── train.py
│   └── inference.py
├── configs/
│   └── config.yaml
├── notebooks/
├── outputs/
└── README.md
```

# 개발 규칙 (Development Rules)

- **모듈화**: 각 컴포넌트는 독립적으로 교체 가능하도록 설계합니다.
- **인터페이스 일관성**: 모든 프로세서, 특징 추출기, 모델은 일관된 인터페이스를 따릅니다. 이는 템플릿으로 관리되어, 팀원들과 협업 시 일관성을 유지할 수 있습니다. 이에 대한 내용은 template.md에 기술되어 있습니다.
- **설정 관리**: 하이퍼파라미터와 설정은 config 파일에서 관리합니다.
- **실험 관리**: 각 실험의 설정, 로그, 결과를 체계적으로 저장합니다.
- **코드 품질**: 
  - PEP 8 스타일 가이드를 준수합니다
  - 적절한 주석과 문서화를 포함합니다
  - 단위 테스트를 작성합니다
- **버전 관리**: Git을 사용하여 코드 변경사항을 관리합니다.
- **재현성**: 난수 시드 설정으로 결과의 재현성을 보장합니다.
