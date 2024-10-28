import torch
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        모델 초기화
        :param input_dim: 입력 차원
        :param output_dim: 출력 차원
        """
        super(CustomModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, output_dim)

        ## 여기 파라미터 128 이거 yaml으로 조정하도록 코드 변경
        ## + cuda 사용해서 gpu 사용하도록 코드 변경

    def forward(self, x):
        """
        순전파 정의
        :param x: 입력 데이터
        :return: 모델 출력
        """
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x
