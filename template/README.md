# Template Directory

이 디렉토리는 머신러닝 프로젝트를 위한 템플릿 코드를 제공합니다. 각 파일은 특정 기능을 수행하는 모듈로 구성되어 있으며, 프로젝트의 구조를 체계적으로 설계하는 데 도움을 줍니다.

## 파일 설명

- **data_template.py**: 데이터 로드 및 전처리, DataLoader 생성 기능을 제공합니다.
- **model_template.py**: PyTorch 기반의 간단한 신경망 모델을 정의합니다.
- **train_template.py**: 모델 훈련 및 평가를 위한 함수들을 포함하고 있습니다.
- **utils_template.py**: 시드 설정, 모델 저장 및 로드와 같은 유틸리티 기능을 제공합니다.
- **main_template.py**: 전체 파이프라인을 실행하는 메인 스크립트로, 명령행 인자를 통해 설정을 받아들입니다.

## 사용 방법

1. 각 템플릿 파일을 기반으로 프로젝트의 요구사항에 맞게 코드를 수정합니다.
2. `main_template.py`를 실행하여 전체 파이프라인을 테스트합니다.
3. 필요에 따라 각 모듈을 확장하거나 수정하여 프로젝트에 맞게 조정합니다.

이 템플릿은 코드의 유지보수를 쉽게 하고, 팀원들과의 협업을 원활하게 할 수 있도록 설계되었습니다.
