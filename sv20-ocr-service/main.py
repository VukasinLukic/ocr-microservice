"""
SV-20 OCR Mikroservis - Entry Point
Glavna skripta za pokretanje OCR servisa.

Pokretanje:
    python main.py

Servis ce biti dostupan na:
    http://127.0.0.1:9001

Dokumentacija API-ja:
    http://127.0.0.1:9001/docs
"""

import uvicorn
import logging
import sys
import argparse
from pathlib import Path

# Dodaj root direktorijum u path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config

# Konfiguracija loggera
def setup_logging():
    """Podesava logging za servis."""
    Config.ensure_directories()

    log_file = Config.LOGS_DIR / 'ocr_service.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger(__name__)


def check_prerequisites(logger) -> bool:
    """
    Proverava da li su svi preduslovi ispunjeni.

    Returns:
        True ako je sve OK, False ako nedostaje nesto
    """
    all_ok = True

    # Proveri models direktorijum
    if not Config.MODELS_DIR.exists():
        logger.warning(f"Models direktorijum ne postoji: {Config.MODELS_DIR}")
        logger.warning("Modeli ce biti preuzeti pri prvom pokretanju OCR-a")
        Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Proveri template
    template_path = Config.TEMPLATES_DIR / "sv20_template.json"
    if not template_path.exists():
        logger.error(f"Template fajl ne postoji: {template_path}")
        logger.error("Kreiraj sv20_template.json pre pokretanja!")
        all_ok = False

    # Proveri logs direktorijum
    if not Config.LOGS_DIR.exists():
        Config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    return all_ok


def print_banner():
    """Stampa banner pri pokretanju."""
    banner = """
    ============================================================
    |                                                          |
    |           SV-20 OCR Mikroservis v1.0                     |
    |                                                          |
    ============================================================
    |                                                          |
    |   Offline OCR servis za prepoznavanje podataka           |
    |   sa SV-20 obrazaca studenata                            |
    |                                                          |
    |   Tehnologije:                                           |
    |   - FastAPI + Uvicorn                                    |
    |   - EasyOCR (sr_cyrl + sr_latn)                          |
    |   - OpenCV za image processing                           |
    |   - AMD DirectML GPU podrska                             |
    |                                                          |
    ============================================================
    """
    print(banner)


def main():
    """Glavna funkcija za pokretanje servisa."""
    # Parse argumenata
    parser = argparse.ArgumentParser(description='SV-20 OCR Mikroservis')
    parser.add_argument('--host', default=Config.HOST,
                       help=f'Host adresa (default: {Config.HOST})')
    parser.add_argument('--port', type=int, default=Config.PORT,
                       help=f'Port (default: {Config.PORT})')
    parser.add_argument('--reload', action='store_true',
                       help='Omoguci auto-reload (za development)')
    parser.add_argument('--workers', type=int, default=1,
                       help='Broj worker procesa (default: 1)')
    parser.add_argument('--log-level', default='info',
                       choices=['debug', 'info', 'warning', 'error'],
                       help='Log level (default: info)')
    args = parser.parse_args()

    # Setup
    print_banner()
    logger = setup_logging()

    # Provera preduslova
    logger.info("Proveravam preduslove...")
    if not check_prerequisites(logger):
        logger.error("Preduslovi nisu ispunjeni. Zaustavljam.")
        sys.exit(1)

    # Detekcija GPU-a
    gpu_backend = Config.detect_gpu()
    logger.info(f"Detektovan GPU backend: {gpu_backend}")
    if gpu_backend == "directml":
        logger.info("AMD DirectML GPU akceleracija je dostupna")
    elif gpu_backend == "cuda":
        logger.info("NVIDIA CUDA GPU akceleracija je dostupna")
    else:
        logger.info("GPU nije dostupan, koristice se CPU")

    # Informacije o pokretanju
    logger.info("=" * 50)
    logger.info(f"Pokrecem server na http://{args.host}:{args.port}")
    logger.info(f"API Dokumentacija: http://{args.host}:{args.port}/docs")
    logger.info(f"Health Check: http://{args.host}:{args.port}/api/health")
    logger.info("=" * 50)

    # Pokreni Uvicorn server
    try:
        uvicorn.run(
            "api.server:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1,
            log_level=args.log_level,
            access_log=True
        )
    except KeyboardInterrupt:
        logger.info("Server zaustavljen od strane korisnika")
    except Exception as e:
        logger.error(f"Greska pri pokretanju servera: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
