"""
SV-20 OCR Mikroservis - Model Downloader
Skripta za preuzimanje svih potrebnih modela unapred.
Pokrenuti jednom pre offline koriscenja.

Pokretanje:
    python download_models.py
"""

import os
import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
MODELS_DIR = Path(__file__).parent / "models"


def print_banner():
    """Stampa banner."""
    print("""
    ============================================================
    |                                                          |
    |         SV-20 OCR - Offline Model Downloader             |
    |                                                          |
    ============================================================
    |                                                          |
    |   Ovaj skript preuzima sve potrebne modele za            |
    |   offline rad OCR servisa.                               |
    |                                                          |
    |   Modeli koji ce biti preuzeti:                          |
    |   - EasyOCR text detection model (~90 MB)                |
    |   - Cyrillic recognition model (~107 MB)                 |
    |   - Latin recognition model (~107 MB)                    |
    |                                                          |
    |   Ukupno: ~300 MB                                        |
    |                                                          |
    ============================================================
    """)


def create_directories():
    """Kreira potrebne direktorijume."""
    dirs = [
        MODELS_DIR,
        Path(__file__).parent / "logs",
        Path(__file__).parent / "templates",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Direktorijum spreman: {d}")


def download_easyocr_models():
    """
    Preuzima EasyOCR modele koristeci EasyOCR biblioteku.
    Preuzima Cyrillic i Latin modele odvojeno jer se ne mogu kombinovati.
    """
    logger.info("=" * 50)
    logger.info("Preuzimanje EasyOCR modela...")
    logger.info("=" * 50)

    try:
        import easyocr
    except ImportError:
        logger.error("EasyOCR nije instaliran!")
        logger.error("Instaliraj sa: pip install easyocr")
        return False

    success = True

    # 1. Preuzmi Cyrillic model (rs_cyrillic + en)
    try:
        logger.info("")
        logger.info("1/2: Preuzimam Serbian Cyrillic model...")
        logger.info("     Ovo moze potrajati nekoliko minuta...")

        reader_cyrillic = easyocr.Reader(
            ['rs_cyrillic', 'en'],
            model_storage_directory=str(MODELS_DIR),
            download_enabled=True,
            gpu=False,
            verbose=False
        )
        logger.info("     [OK] Serbian Cyrillic model preuzet!")
        del reader_cyrillic

    except Exception as e:
        logger.error(f"     [GRESKA] Cyrillic model: {e}")
        success = False

    # 2. Preuzmi Latin model (rs_latin + en)
    try:
        logger.info("")
        logger.info("2/2: Preuzimam Serbian Latin model...")
        logger.info("     Ovo moze potrajati nekoliko minuta...")

        reader_latin = easyocr.Reader(
            ['rs_latin', 'en'],
            model_storage_directory=str(MODELS_DIR),
            download_enabled=True,
            gpu=False,
            verbose=False
        )
        logger.info("     [OK] Serbian Latin model preuzet!")
        del reader_latin

    except Exception as e:
        logger.error(f"     [GRESKA] Latin model: {e}")
        success = False

    if success:
        logger.info("")
        logger.info("Svi modeli uspesno preuzeti!")

    return success


def verify_models():
    """Verifikuje da su svi modeli preuzeti."""
    logger.info("")
    logger.info("=" * 50)
    logger.info("Verifikacija modela...")
    logger.info("=" * 50)

    expected_files = [
        "craft_mlt_25k.pth",
        "cyrillic_g2.pth",
        "latin_g2.pth",
    ]

    all_ok = True
    total_size = 0

    for filename in expected_files:
        filepath = MODELS_DIR / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            total_size += size_mb
            logger.info(f"  [OK] {filename} ({size_mb:.1f} MB)")
        else:
            found = False
            for subpath in MODELS_DIR.rglob(filename):
                size_mb = subpath.stat().st_size / (1024 * 1024)
                total_size += size_mb
                logger.info(f"  [OK] {filename} ({size_mb:.1f} MB)")
                found = True
                break
            if not found:
                logger.warning(f"  [!] {filename} - nije pronadjen (mozda ima drugi naziv)")

    logger.info(f"\nUkupna velicina modela: {total_size:.1f} MB")
    return all_ok


def test_ocr():
    """Testira OCR sa jednostavnim primerom."""
    logger.info("")
    logger.info("=" * 50)
    logger.info("Testiranje OCR Engine-a...")
    logger.info("=" * 50)

    try:
        import easyocr
        import numpy as np

        # Test Cyrillic reader
        logger.info("  Testiram Cyrillic reader...")
        reader = easyocr.Reader(
            ['rs_cyrillic', 'en'],
            model_storage_directory=str(MODELS_DIR),
            download_enabled=False,
            gpu=False,
            verbose=False
        )
        logger.info("  [OK] Cyrillic reader inicijalizovan!")
        del reader

        # Test Latin reader
        logger.info("  Testiram Latin reader...")
        reader = easyocr.Reader(
            ['rs_latin', 'en'],
            model_storage_directory=str(MODELS_DIR),
            download_enabled=False,
            gpu=False,
            verbose=False
        )
        logger.info("  [OK] Latin reader inicijalizovan!")
        del reader

        return True

    except Exception as e:
        logger.error(f"  [GRESKA] Test nije uspeo: {e}")
        return False


def check_gpu():
    """Proverava dostupnost GPU-a."""
    logger.info("")
    logger.info("=" * 50)
    logger.info("Provera GPU podrske...")
    logger.info("=" * 50)

    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        logger.info(f"Dostupni ONNX provideri: {providers}")

        if 'DmlExecutionProvider' in providers:
            logger.info("  [OK] AMD DirectML GPU je dostupan!")
            return "directml"
        elif 'CUDAExecutionProvider' in providers:
            logger.info("  [OK] NVIDIA CUDA GPU je dostupan!")
            return "cuda"
        else:
            logger.info("  [INFO] GPU nije dostupan, koristice se CPU")
            return "cpu"
    except ImportError:
        logger.warning("  [!] onnxruntime nije instaliran")
        return "cpu"


def main():
    """Glavna funkcija."""
    print_banner()

    create_directories()
    gpu_backend = check_gpu()

    if not download_easyocr_models():
        logger.error("")
        logger.error("Preuzimanje nekih modela nije uspelo!")
        logger.error("Proveri internet konekciju i pokusaj ponovo.")
        sys.exit(1)

    verify_models()

    if test_ocr():
        print("""
    ============================================================
    |                                                          |
    |               INSTALACIJA USPESNA!                       |
    |                                                          |
    ============================================================
    |                                                          |
    |   Svi modeli su preuzeti i verifikovani.                 |
    |   Servis je spreman za offline rad.                      |
    |                                                          |
    |   Pokrenite servis sa:                                   |
    |                                                          |
    |       python main.py                                     |
    |                                                          |
    |   API dokumentacija ce biti dostupna na:                 |
    |                                                          |
    |       http://127.0.0.1:9001/docs                         |
    |                                                          |
    ============================================================
        """)
    else:
        print("""
    ============================================================
    |                                                          |
    |            INSTALACIJA ZAVRSENA SA UPOZORENJIMA          |
    |                                                          |
    ============================================================
    |                                                          |
    |   Modeli su preuzeti, ali test nije prosao.              |
    |   Servis moze raditi, ali proverite konfiguraciju.       |
    |                                                          |
    ============================================================
        """)


if __name__ == "__main__":
    main()
