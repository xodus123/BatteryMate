"""Config 로더 - YAML 파일 로딩"""
import yaml
from pathlib import Path
from typing import Dict


class ConfigLoader:
    """YAML Config 로더"""

    @staticmethod
    def load(config_name: str) -> Dict:
        """
        Config 파일 로드

        Args:
            config_name: 'cnn', 'autoencoder', 'evaluation', 'logging'

        Returns:
            config dict
        """
        config_path = Path(__file__).parent / f'{config_name}.yaml'

        if not config_path.exists():
            raise FileNotFoundError(f"Config 파일을 찾을 수 없습니다: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        print(f"✅ Config 로드 완료: {config_name}.yaml")
        return config


# 사용 예시
if __name__ == "__main__":
    config = ConfigLoader.load('cnn')
    print(config)
