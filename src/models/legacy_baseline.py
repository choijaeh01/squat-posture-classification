#!/usr/bin/env python3
"""
베이스라인 모델 구현
1. CNN-GRU (KU-HAR 논문 기반)
2. 1D CNN (간단한 베이스라인)
"""

import torch
import torch.nn as nn


# ============================================================================
# 1. CNN-GRU 모델 (Transfer Learning용 베이스라인)
# ============================================================================

class CNNGRU(nn.Module):
    """
    CNN-GRU 모델 (KU-HAR 사전학습 모델과 동일한 구조)
    
    입력: [batch, channels=18, timesteps=400]
    출력: [batch, num_classes=5]
    
    구조:
    - CNN: 특징 추출 (spatial features)
    - GRU: 시간적 패턴 학습 (temporal features)
    - FC: 분류
    """
    
    def __init__(self, 
                 input_channels=18,
                 num_classes=5,
                 cnn_channels=[64, 128, 256],
                 gru_hidden=128,
                 gru_layers=2,
                 dropout=0.5):
        """
        Args:
            input_channels: 입력 채널 수 (18 = 3 sensors × 6 axes)
            num_classes: 클래스 수 (5)
            cnn_channels: CNN 각 레이어 채널 수
            gru_hidden: GRU hidden size
            gru_layers: GRU 레이어 수
            dropout: Dropout 비율
        """
        super(CNNGRU, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, cnn_channels[0], kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels[0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(cnn_channels[0], cnn_channels[1], kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels[1]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(cnn_channels[1], cnn_channels[2], kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels[2]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout)
        )
        
        # GRU layers
        # 입력: [batch, seq_len, features]
        # seq_len = 400 / 2 / 2 / 2 = 50
        self.gru = nn.GRU(
            input_size=cnn_channels[2],
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0,
            bidirectional=True  # Bidirectional for better performance
        )
        
        self.feature_dim = gru_hidden * 2  # *2 for bidirectional

        # FC layers
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward_features(self, x):
        """
        Forward pass up to pooled features.
        
        Args:
            x: [batch, channels=18, timesteps=400]
        
        Returns:
            features: [batch, feature_dim]
        """
        # CNN feature extraction
        x = self.conv1(x)  # [batch, 64, 200]
        x = self.conv2(x)  # [batch, 128, 100]
        x = self.conv3(x)  # [batch, 256, 50]
        
        # Reshape for GRU: [batch, channels, timesteps] → [batch, timesteps, channels]
        x = x.transpose(1, 2)  # [batch, 50, 256]
        
        # GRU
        x, _ = self.gru(x)  # [batch, 50, 256] (128*2 bidirectional)
        
        # Global average pooling over time
        features = x.mean(dim=1)  # [batch, 256]
        return features
    
    def forward(self, x, return_features=False):
        """
        Full forward pass with optional feature return.
        """
        features = self.forward_features(x)
        
        # Classification
        logits = self.fc(features)  # [batch, num_classes]
        
        if return_features:
            return logits, features
        return logits


# ============================================================================
# 2. Simple 1D CNN (빠른 베이스라인)
# ============================================================================

class SimpleCNN(nn.Module):
    """
    간단한 1D CNN 모델
    
    입력: [batch, channels=18, timesteps=400]
    출력: [batch, num_classes=5]
    """
    
    def __init__(self, 
                 input_channels=18,
                 num_classes=5,
                 dropout=0.5):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            # Conv block 2
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            # Conv block 3
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            # Conv block 4
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Global average pooling
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch, channels=18, timesteps=400]
        Returns:
            logits: [batch, num_classes=5]
        """
        x = self.features(x)
        logits = self.classifier(x)
        return logits


# ============================================================================
# 3. Residual 1D CNN (더 깊은 모델)
# ============================================================================

class ResidualBlock(nn.Module):
    """1D CNN용 Residual Block"""
    
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.3):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet1D(nn.Module):
    """1D ResNet for time series classification"""
    
    def __init__(self,
                 input_channels=18,
                 num_classes=5,
                 layers=[2, 2, 2, 2],
                 dropout=0.3):
        super(ResNet1D, self).__init__()
        
        self.in_channels = 64
        
        # Initial conv layer
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual layers
        self.layer1 = self._make_layer(64, layers[0], stride=1, dropout=dropout)
        self.layer2 = self._make_layer(128, layers[1], stride=2, dropout=dropout)
        self.layer3 = self._make_layer(256, layers[2], stride=2, dropout=dropout)
        self.layer4 = self._make_layer(512, layers[3], stride=2, dropout=dropout)
        
        # Global average pooling + FC
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, out_channels, blocks, stride, dropout):
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride, dropout))
        self.in_channels = out_channels
        
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, dropout=dropout))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


# ============================================================================
# 모델 생성 함수
# ============================================================================

def create_model(model_name='cnngru', num_classes=5, **kwargs):
    """
    모델 생성 헬퍼 함수
    
    Args:
        model_name: 'cnngru', 'simple_cnn', 'resnet1d'
        num_classes: 클래스 수
        **kwargs: 모델별 추가 파라미터
    
    Returns:
        model: PyTorch 모델
    """
    if model_name == 'cnngru':
        return CNNGRU(num_classes=num_classes, **kwargs)
    elif model_name == 'simple_cnn':
        return SimpleCNN(num_classes=num_classes, **kwargs)
    elif model_name == 'resnet1d':
        return ResNet1D(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def count_parameters(model):
    """모델 파라미터 수 계산"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# 테스트
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 모델 테스트")
    print("=" * 60)
    
    batch_size = 8
    channels = 18
    timesteps = 400
    num_classes = 5
    
    # 더미 입력
    x = torch.randn(batch_size, channels, timesteps)
    
    # 모델 테스트
    models = {
        'CNN-GRU': CNNGRU(),
        'Simple CNN': SimpleCNN(),
        'ResNet1D': ResNet1D()
    }
    
    for name, model in models.items():
        print(f"\n{name}:")
        print(f"   파라미터 수: {count_parameters(model):,}")
        
        # Forward pass
        output = model(x)
        print(f"   입력 shape: {x.shape}")
        print(f"   출력 shape: {output.shape}")
        
        assert output.shape == (batch_size, num_classes), \
            f"Output shape mismatch: {output.shape}"
    
    print("\n✅ 모든 모델 테스트 통과!")
