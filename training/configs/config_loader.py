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
            config_name: 파일 이름 ('cnn') 또는 전체 경로 ('training/configs/cnn.yaml')

        Returns:
            config dict
        """
        # 전체 경로인지 확인
        if config_name.endswith('.yaml') or '/' in config_name:
            config_path = Path(config_name)
            # 상대 경로면 프로젝트 루트 기준으로 변환
            if not config_path.is_absolute():
                project_root = Path(__file__).parent.parent.parent
                config_path = project_root / config_name
        else:
            # 이름만 주어진 경우 기존 방식
            config_path = Path(__file__).parent / f'{config_name}.yaml'

        if not config_path.exists():
            raise FileNotFoundError(f"Config 파일을 찾을 수 없습니다: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        print(f"✅ Config 로드 완료: {config_path.name}")
        return config


# 사용 예시
if __name__ == "__main__":
    config = ConfigLoader.load('cnn')
    print(config)
