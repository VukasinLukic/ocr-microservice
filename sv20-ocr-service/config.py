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

    # WS6 - TrOCR rukom pisan tekst (EKSPERIMENTALNO, videti
    # processors/handwriting_ocr.py i ANALIZA-OCR-SERVISA.md). Default False:
    # kad je isključeno, `transformers` paket se čak ni ne pokušava importovati
    # niti se model preuzima - servis radi identično kao pre WS6.
    ENABLE_HANDWRITING_TROCR = False
    # Ako je EasyOCR confidence za TEXT/ALPHANUMERIC polje ISPOD ovog praga,
    # I ENABLE_HANDWRITING_TROCR je uključen, probaj TrOCR kao drugo mišljenje
    # (pored polja eksplicitno označenih "handwriting": true u template-u).
    HANDWRITING_FALLBACK_CONFIDENCE = 0.5

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
        Detektuje backend koji STVARNO ubrzava OCR.

        VAŽNO (ANALIZA-OCR-SERVISA.md 4.3): EasyOCR interno koristi PyTorch,
        NE onnxruntime. Ranija verzija ove funkcije proveravala je
        `onnxruntime.get_available_providers()` za DirectML - to je bio
        potpuno nepovezan signal od onoga što EasyOCR stvarno koristi za
        inferencu, pa je servis lagao da radi na AMD DirectML GPU-u dok je u
        stvarnosti uvek radio na CPU-u. Ova verzija proverava ono što je
        stvarno relevantno: `torch.cuda.is_available()` (NVIDIA) ili prisustvo
        `torch_directml` paketa (AMD/Intel preko DirectML - mora biti
        eksplicitno instaliran, trenutno NIJE deo requirements.txt jer
        zahteva dodatne izmene u OCR engine kodu da bi se tenzori stvarno
        stavili na taj device - vidi napomenu u ocr_engine.py).

        Returns:
            str: "cuda" (NVIDIA), "directml" (AMD/Intel preko torch_directml),
                 ili "cpu" (stvarno stanje na većini AMD/Windows mašina danas)
        """
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        except Exception:
            pass

        try:
            import torch_directml  # noqa: F401 - samo provera da li je instaliran
            return "directml"
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
