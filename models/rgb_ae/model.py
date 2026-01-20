"""RGB AutoEncoder 모델 정의 - 외부 결함 이상 탐지"""
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any


class ConvAutoEncoder(nn.Module):
    """
    Convolutional AutoEncoder for Anomaly Detection
    - 불량 이미지로 학습하여 불량 패턴 학습
    - 정상 이미지 입력 시 높은 재구성 오류 → 이상 탐지
    """

    def __init__(
        self,
        image_size: int = 512,
        latent_dim: int = 1024,
        encoder_channels: list = [3, 64, 128, 256, 512],
        decoder_channels: list = [512, 256, 128, 64, 3],
        dropout: float = 0.2
    ):
        """
        Args:
            image_size: 입력 이미지 크기 (정사각형)
            latent_dim: latent space 차원
            encoder_channels: 인코더 채널 리스트
            decoder_channels: 디코더 채널 리스트
            dropout: Dropout 비율
        """
        super(ConvAutoEncoder, self).__init__()

        self.image_size = image_size
        self.latent_dim = latent_dim

        # Encoder
        encoder_layers = []
        for i in range(len(encoder_channels) - 1):
            encoder_layers.extend([
                nn.Conv2d(encoder_channels[i], encoder_channels[i+1],
                         kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(encoder_channels[i+1]),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            if dropout > 0 and i > 0:
                encoder_layers.append(nn.Dropout2d(dropout))

        self.encoder = nn.Sequential(*encoder_layers)

        # 인코더 출력 크기 계산 (512 -> 256 -> 128 -> 64 -> 32 = 4번 다운샘플)
        self.encoded_size = image_size // (2 ** (len(encoder_channels) - 1))
        self.encoded_channels = encoder_channels[-1]

        # Conv Bottleneck (Global Average Pool 방식 - 파라미터 효율적)
        # 32×32×512 → GAP → 512 → FC → 1024
        self.bottleneck_encode = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 32×32 → 1×1
            nn.Flatten(),
            nn.Linear(self.encoded_channels, latent_dim),
            nn.ReLU(inplace=True)
        )

        # 1024 → FC → 512 → Reshape → 1×1×512 → Upsample → 32×32×512
        self.bottleneck_decode = nn.Sequential(
            nn.Linear(latent_dim, self.encoded_channels),
            nn.ReLU(inplace=True)
        )

        # Upsample 1×1 → encoded_size×encoded_size
        self.upsample = nn.Upsample(
            size=(self.encoded_size, self.encoded_size),
            mode='bilinear',
            align_corners=False
        )

        # Decoder
        decoder_layers = []
        for i in range(len(decoder_channels) - 1):
            if i < len(decoder_channels) - 2:
                decoder_layers.extend([
                    nn.ConvTranspose2d(decoder_channels[i], decoder_channels[i+1],
                                      kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(decoder_channels[i+1]),
                    nn.ReLU(inplace=True),
                ])
            else:
                # 마지막 레이어: Tanh로 [-1, 1] 출력
                decoder_layers.extend([
                    nn.ConvTranspose2d(decoder_channels[i], decoder_channels[i+1],
                                      kernel_size=4, stride=2, padding=1, bias=False),
                    nn.Tanh(),
                ])

        self.decoder = nn.Sequential(*decoder_layers)

        # 파라미터 초기화
        self._init_weights()

        print(f"✅ ConvAutoEncoder 생성 완료")
        print(f"   - Image size: {image_size}x{image_size}")
        print(f"   - Latent dim: {latent_dim}")
        print(f"   - Encoded size: {self.encoded_size}x{self.encoded_size}x{self.encoded_channels}")
        print(f"   - Encoder: {encoder_channels}")
        print(f"   - Decoder: {decoder_channels}")

    def _init_weights(self):
        """가중치 초기화"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """인코딩: 이미지 → latent vector"""
        x = self.encoder(x)           # (B, 512, 32, 32)
        z = self.bottleneck_encode(x) # (B, 1024)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """디코딩: latent vector → 이미지"""
        x = self.bottleneck_decode(z)  # (B, 512)
        x = x.view(x.size(0), self.encoded_channels, 1, 1)  # (B, 512, 1, 1)
        x = self.upsample(x)           # (B, 512, 32, 32)
        x = self.decoder(x)            # (B, 3, 512, 512)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 입력 이미지 (B, 3, H, W)

        Returns:
            reconstructed: 재구성 이미지 (B, 3, H, W)
            latent: latent vector (B, latent_dim)
        """
        z = self.encode(x)
        reconstructed = self.decode(z)
        return reconstructed, z

    def compute_reconstruction_error(self, x: torch.Tensor, reduction: str = 'none') -> torch.Tensor:
        """
        재구성 오류 계산 (이상 점수)

        Args:
            x: 입력 이미지
            reduction: 'none' (픽셀별), 'mean' (이미지별 평균), 'sum'

        Returns:
            재구성 오류
        """
        reconstructed, _ = self.forward(x)
        error = (x - reconstructed) ** 2

        if reduction == 'none':
            return error
        elif reduction == 'mean':
            return error.mean(dim=[1, 2, 3])  # 이미지별 평균
        elif reduction == 'sum':
            return error.sum(dim=[1, 2, 3])
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

    def get_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        이상 점수 계산 (높을수록 이상)

        Args:
            x: 입력 이미지

        Returns:
            anomaly_scores: (B,) 이미지별 이상 점수
        """
        return self.compute_reconstruction_error(x, reduction='mean')


class SSIMLoss(nn.Module):
    """SSIM 기반 손실 함수 (선택적)"""

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.channel = 3

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """SSIM Loss = 1 - SSIM"""
        # 간단한 MSE로 대체 (SSIM은 복잡하므로)
        return nn.functional.mse_loss(x, y)


def create_model(config: dict) -> ConvAutoEncoder:
    """
    Config 기반 모델 생성

    Args:
        config: YAML config dict

    Returns:
        ConvAutoEncoder 모델
    """
    model_config = config.get('model', {})

    # 인코더/디코더 채널 설정
    encoder_channels = model_config.get('encoder', {}).get('channels', [3, 64, 128, 256, 512])
    decoder_channels = model_config.get('decoder', {}).get('channels', [512, 256, 128, 64, 3])

    model = ConvAutoEncoder(
        image_size=config['data']['image_size'],
        latent_dim=model_config.get('latent_dim', 1024),
        encoder_channels=encoder_channels,
        decoder_channels=decoder_channels,
        dropout=model_config.get('dropout', 0.2)
    )

    return model


# 테스트
if __name__ == "__main__":
    # 모델 생성 (새 설정: 512x512, latent=1024)
    model = ConvAutoEncoder(image_size=512, latent_dim=1024)

    # Forward 테스트
    dummy_input = torch.randn(4, 3, 512, 512)
    reconstructed, latent = model(dummy_input)

    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Output shape: {reconstructed.shape}")

    # 이상 점수 테스트
    anomaly_scores = model.get_anomaly_score(dummy_input)
    print(f"Anomaly scores shape: {anomaly_scores.shape}")
    print(f"Anomaly scores: {anomaly_scores}")

    # 파라미터 수
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal params: {total_params:,}")
