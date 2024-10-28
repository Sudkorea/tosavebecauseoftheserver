import torch
import torch.nn as nn
from ._trainer import Trainer

class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, args, data):
        """
        Neural Collaborative Filtering 모델
        Args:
            args: 모델 설정값
            data: 데이터 정보를 담은 딕셔너리
        """
        super().__init__()
        
        # 데이터 크기 정보
        self.field_dims = data['field_dims']
        self.user_field_idx = [0]
        self.item_field_idx = [1]
        
        # 모델 하이퍼파라미터
        self.embed_dim = args.embed_dim
        self.mlp_dims = args.mlp_dims
        self.dropout = args.dropout
        self.batchnorm = args.batchnorm
        
        # GMF를 위한 임베딩
        self.user_gmf_embedding = nn.Embedding(self.field_dims[0], self.embed_dim)
        self.item_gmf_embedding = nn.Embedding(self.field_dims[1], self.embed_dim)
        
        # MLP를 위한 임베딩
        self.user_mlp_embedding = nn.Embedding(self.field_dims[0], self.embed_dim)
        self.item_mlp_embedding = nn.Embedding(self.field_dims[1], self.embed_dim)
        
        # MLP 레이어 구성
        self.mlp_layers = []
        input_dim = self.embed_dim * 2
        
        for i, (in_size, out_size) in enumerate(zip([input_dim] + self.mlp_dims[:-1], self.mlp_dims)):
            self.mlp_layers.extend([
                nn.Linear(in_size, out_size),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ])
            if self.batchnorm:
                self.mlp_layers.append(nn.BatchNorm1d(out_size))
                
        self.mlp = nn.Sequential(*self.mlp_layers)
        
        # 최종 예측 레이어
        self.fc = nn.Linear(self.mlp_dims[-1] + self.embed_dim, 1)
        
        # 가중치 초기화
        self._initialize_weights()
        
    def _initialize_weights(self):
        """가중치 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        """
        순전파
        Args:
            x: 입력 텐서 (batch_size, 2) - user_indices와 item_indices를 포함
        Returns:
            예측된 평점
        """
        user_indices = x[:, self.user_field_idx].squeeze(1)
        item_indices = x[:, self.item_field_idx].squeeze(1)
        
        # GMF 부분
        user_gmf_embed = self.user_gmf_embedding(user_indices)
        item_gmf_embed = self.item_gmf_embedding(item_indices)
        gmf = user_gmf_embed * item_gmf_embed
        
        # MLP 부분
        user_mlp_embed = self.user_mlp_embedding(user_indices)
        item_mlp_embed = self.item_mlp_embedding(item_indices)
        mlp_input = torch.cat([user_mlp_embed, item_mlp_embed], dim=1)
        mlp_output = self.mlp(mlp_input)
        
        # 결합 및 최종 예측
        x = torch.cat([gmf, mlp_output], dim=1)
        x = self.fc(x).squeeze(1)
        
        return x

    def fit(self, train_loader, valid_loader, config):
        """
        모델 학습을 수행합니다.
        Args:
            train_loader: 학습 데이터 로더
            valid_loader: 검증 데이터 로더
            config: 학습 설정
        Returns:
            best_loss: 최적의 검증 손실값
        """
        # 손실 함수 설정
        criterion = getattr(nn, config.loss)()
        
        # 옵티마이저 설정
        optimizer = getattr(torch.optim, config.optimizer.type)(
            self.parameters(),
            **config.optimizer.args
        )
        
        # 스케줄러 설정
        scheduler = None
        if config.lr_scheduler.use:
            scheduler = getattr(torch.optim.lr_scheduler, config.lr_scheduler.type)(
                optimizer,
                **config.lr_scheduler.args
            )
        
        # 트레이너 초기화 및 학습 수행
        trainer = Trainer(
            model=self,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=config.device
        )
        
        return trainer.train(train_loader, valid_loader, config)
    
    def predict(self, test_loader, config):
        """
        테스트 데이터에 대한 예측을 수행합니다.
        Args:
            test_loader: 테스트 데이터 로더
            config: 설정
        Returns:
            predictions: 예측값 배열
        """
        trainer = Trainer(
            model=self,
            criterion=None,
            optimizer=None,
            device=config.device
        )
        return trainer.predict(test_loader)
