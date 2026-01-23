"""
SV-20 OCR Mikroservis - Konfiguracija
Konfiguracione postavke za server, OCR engine i image processing.
"""

import os
from pathlib import Path


class Config:
    """Centralna konfiguracija za OCR servis."""

    # Server
    HOST = "127.0.0.1"
    PORT = 9001

    # Paths
    BASE_DIR = Path(__file__).parent
    MODELS_DIR = BASE_DIR / "models"
    TEMPLATES_DIR = BASE_DIR / "templates"
    LOGS_DIR = BASE_DIR / "logs"

    # OCR Settings
    # Serbian Cyrillic and Latin - EasyOCR requires separate readers
    # Cyrillic script (includes Serbian Cyrillic)
    OCR_LANGUAGES_CYRILLIC = ['rs_cyrillic', 'en']
    # Latin script (includes Serbian Latin)
    OCR_LANGUAGES_LATIN = ['rs_latin', 'en']
    # Default to Latin for most forms
    OCR_LANGUAGES = ['rs_cyrillic', 'rs_latin', 'en']
    CONFIDENCE_THRESHOLD = 0.5

    # Hardware Detection
    USE_GPU = True
    GPU_BACKEND = "directml"  # AMD Radeon

    # Image Processing
    TARGET_DPI = 300
    BINARIZATION_THRESHOLD = 127

    # Validation
    JMBG_LENGTH = 13
    INDEX_PATTERN = r"^\d{4}/\d{4}$"
    DATE_FORMATS = ["%d.%m.%Y", "%d/%m/%Y", "%Y-%m-%d"]

    # Database (opciono - ako se koristi MySQL)
    DB_CONFIG = {
        "host": "localhost",
        "port": 3306,
        "database": "bazaocr",
        "user": "root",
        "password": ""
    }

    @classmethod
    def detect_gpu(cls) -> str:
        """
        Automatska detekcija GPU-a.

        Returns:
            str: "directml" za AMD, "cuda" za NVIDIA, ili "cpu"
        """
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if 'DmlExecutionProvider' in providers:
                return "directml"
            elif 'CUDAExecutionProvider' in providers:
                return "cuda"
        except ImportError:
            pass
        except Exception:
            pass
        return "cpu"

    @classmethod
    def ensure_directories(cls):
        """Kreira potrebne direktorijume ako ne postoje."""
        for directory in [cls.MODELS_DIR, cls.TEMPLATES_DIR, cls.LOGS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
