import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(self, args, data):
        """
        Neural Collaborative Filtering 모델
        Args:
            args: 모델 설정값
            data: 데이터 정보를 담은 딕셔너리
        """
        super().__init__()
        
        # 데이터 크기 정보
        self.n_users = len(data['user_map'])
        self.n_items = len(data['book_map'])
        
        # 모델 하이퍼파라미터
        self.embed_dim = args.embed_dim
        self.mlp_dims = args.mlp_dims
        self.dropout = args.dropout
        
        # GMF를 위한 임베딩
        self.user_gmf_embedding = nn.Embedding(self.n_users, self.embed_dim)
        self.item_gmf_embedding = nn.Embedding(self.n_items, self.embed_dim)
        
        # MLP를 위한 임베딩
        self.user_mlp_embedding = nn.Embedding(self.n_users, self.embed_dim)
        self.item_mlp_embedding = nn.Embedding(self.n_items, self.embed_dim)
        
        # MLP 레이어
        self.mlp_layers = []
        input_dim = self.embed_dim * 2
        
        for i, (in_size, out_size) in enumerate(zip([input_dim] + self.mlp_dims[:-1], self.mlp_dims)):
            self.mlp_layers.extend([
                nn.Linear(in_size, out_size),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ])
            if args.batchnorm:
                self.mlp_layers.append(nn.BatchNorm1d(out_size))
                
        self.mlp = nn.Sequential(*self.mlp_layers)
        
        # 최종 예측 레이어
        self.final = nn.Linear(self.mlp_dims[-1] + self.embed_dim, 1)
        
        # 가중치 초기화
        self._init_weights()
        
    def _init_weights(self):
        """가중치 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
                
    def forward(self, user_indices, item_indices):
        """
        순전파
        Args:
            user_indices: 사용자 인덱스 텐서
            item_indices: 아이템 인덱스 텐서
        Returns:
            예측된 평점
        """
        # GMF 부분
        user_gmf_embed = self.user_gmf_embedding(user_indices)
        item_gmf_embed = self.item_gmf_embedding(item_indices)
        gmf_output = user_gmf_embed * item_gmf_embed
        
        # MLP 부분
        user_mlp_embed = self.user_mlp_embedding(user_indices)
        item_mlp_embed = self.item_mlp_embedding(item_indices)
        mlp_input = torch.cat([user_mlp_embed, item_mlp_embed], dim=-1)
        mlp_output = self.mlp(mlp_input)
        
        # 결합
        concat = torch.cat([gmf_output, mlp_output], dim=-1)
        
        # 최종 예측
        prediction = self.final(concat)
        return prediction.squeeze()
