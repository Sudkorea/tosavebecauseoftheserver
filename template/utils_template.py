import os
import random
import numpy as np
import torch

def set_seed(seed):
    """
    시드 설정
    :param seed: 시드 값
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(model, path):
    """
    모델 체크포인트 저장
    :param model: 저장할 모델
    :param path: 저장 경로
    """
    torch.save(model.state_dict(), path)

def load_checkpoint(model, path):
    """
    모델 체크포인트 로드
    :param model: 로드할 모델
    :param path: 로드 경로
    """
    model.load_state_dict(torch.load(path))
