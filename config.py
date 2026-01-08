"""중앙 집중화된 설정 관리 모듈

모든 환경 변수와 하드코딩된 설정값들을 이 파일에서 관리합니다.
실제 값은 .env 파일에 설정하고, 이 파일에서 로드합니다.

사용법:
    from config import settings
    api_key = settings.GEMINI_API_KEY
    model_size = settings.VLM_MODEL_SIZE
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


# .env 파일 로드 (python-dotenv 없이 직접 구현)
def _load_env_file(env_path: str = None) -> dict:
    """환경 변수 파일 로드"""
    if env_path is None:
        env_path = Path(__file__).parent / '.env'

    env_vars = {}

    if Path(env_path).exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # 빈 줄이나 주석 건너뛰기
                if not line or line.startswith('#'):
                    continue
                # KEY=VALUE 형식 파싱
                if '=' in line:
                    key, _, value = line.partition('=')
                    key = key.strip()
                    value = value.strip()
                    # 따옴표 제거
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    env_vars[key] = value

    return env_vars


# 환경 파일 로드 및 환경 변수에 설정
_env_vars = _load_env_file()
for key, value in _env_vars.items():
    if key not in os.environ:  # 기존 환경 변수 우선
        os.environ[key] = value


@dataclass
class Settings:
    """프로젝트 설정"""

    # ===== 프로젝트 경로 =====
    PROJECT_ROOT: Path = field(default_factory=lambda: Path(__file__).parent.absolute())

    # ===== API 키 =====
    GEMINI_API_KEY: Optional[str] = field(
        default_factory=lambda: os.getenv('GEMINI_API_KEY')
    )
    OPENAI_API_KEY: Optional[str] = field(
        default_factory=lambda: os.getenv('OPENAI_API_KEY')
    )

    # ===== VLM 설정 =====
    VLM_MODEL_SIZE: str = field(
        default_factory=lambda: os.getenv('VLM_MODEL_SIZE', '2b')
    )
    VLM_DEFAULT_MODEL: str = field(
        default_factory=lambda: os.getenv('VLM_DEFAULT_MODEL', 'qwen2vl')
    )
    GEMINI_MODEL_NAME: str = field(
        default_factory=lambda: os.getenv('GEMINI_MODEL_NAME', 'gemini-2.0-flash')
    )

    # ===== VLG 설정 =====
    VLG_DEFAULT_MODEL: str = field(
        default_factory=lambda: os.getenv('VLG_DEFAULT_MODEL', 'groundingdino')
    )
    VLG_BOX_THRESHOLD: float = field(
        default_factory=lambda: float(os.getenv('VLG_BOX_THRESHOLD', '0.3'))
    )
    VLG_TEXT_THRESHOLD: float = field(
        default_factory=lambda: float(os.getenv('VLG_TEXT_THRESHOLD', '0.25'))
    )

    # ===== CNN 설정 =====
    CNN_NUM_CLASSES: int = field(
        default_factory=lambda: int(os.getenv('CNN_NUM_CLASSES', '5'))
    )
    CNN_IMAGE_SIZE: int = field(
        default_factory=lambda: int(os.getenv('CNN_IMAGE_SIZE', '512'))
    )

    # ===== 체크포인트 경로 =====
    CNN_CHECKPOINT_PATH: Optional[str] = field(
        default_factory=lambda: os.getenv('CNN_CHECKPOINT_PATH')
    )
    AE_CHECKPOINT_PATH: Optional[str] = field(
        default_factory=lambda: os.getenv('AE_CHECKPOINT_PATH')
    )

    # ===== 웹앱 설정 =====
    WEBAPP_PORT: int = field(
        default_factory=lambda: int(os.getenv('WEBAPP_PORT', '8501'))
    )
    WEBAPP_DEBUG: bool = field(
        default_factory=lambda: os.getenv('WEBAPP_DEBUG', 'false').lower() == 'true'
    )

    # ===== 데이터 경로 =====
    DATA_ROOT: Optional[str] = field(
        default_factory=lambda: os.getenv('DATA_ROOT')
    )

    def __post_init__(self):
        """초기화 후 검증"""
        # API 키 경고
        if not self.GEMINI_API_KEY:
            print("⚠️  GEMINI_API_KEY가 설정되지 않았습니다. Gemini API를 사용하려면 설정하세요.")

    def validate_gemini_api(self) -> bool:
        """Gemini API 키 유효성 검증"""
        return bool(self.GEMINI_API_KEY)

    def get_checkpoint_path(self, model_type: str) -> Optional[Path]:
        """모델별 체크포인트 경로 반환"""
        paths = {
            'cnn': self.CNN_CHECKPOINT_PATH,
            'ae': self.AE_CHECKPOINT_PATH,
        }
        path = paths.get(model_type)
        if path:
            return Path(path)
        return None


# 싱글톤 인스턴스
settings = Settings()


# 테스트 및 디버깅용
if __name__ == "__main__":
    print("=== 현재 설정 ===")
    print(f"PROJECT_ROOT: {settings.PROJECT_ROOT}")
    print(f"GEMINI_API_KEY: {'설정됨' if settings.GEMINI_API_KEY else '미설정'}")
    print(f"VLM_MODEL_SIZE: {settings.VLM_MODEL_SIZE}")
    print(f"VLM_DEFAULT_MODEL: {settings.VLM_DEFAULT_MODEL}")
    print(f"GEMINI_MODEL_NAME: {settings.GEMINI_MODEL_NAME}")
    print(f"VLG_DEFAULT_MODEL: {settings.VLG_DEFAULT_MODEL}")
    print(f"VLG_BOX_THRESHOLD: {settings.VLG_BOX_THRESHOLD}")
    print(f"CNN_NUM_CLASSES: {settings.CNN_NUM_CLASSES}")
    print(f"WEBAPP_PORT: {settings.WEBAPP_PORT}")
