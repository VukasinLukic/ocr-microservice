"""
SV-20 OCR Mikroservis - OCR Engine
EasyOCR wrapper sa podrskom za AMD DirectML i offline rad.
Ucitava srpsku cirilicu i latinicu istovremeno.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class OCREngine:
    """
    EasyOCR wrapper sa podrskom za AMD DirectML i offline rad.
    Ucitava srpsku cirilicu i latinicu istovremeno.
    """

    def __init__(self,
                 languages: List[str] = None,
                 model_storage_directory: str = None,
                 use_gpu: bool = True,
                 gpu_backend: str = "directml"):
        """
        Inicijalizacija OCR engine-a.

        Args:
            languages: Lista jezika za prepoznavanje (default: ['rs_cyrillic', 'rs_latin'])
            model_storage_directory: Putanja do offline modela
            use_gpu: Da li koristiti GPU
            gpu_backend: "directml" za AMD, "cuda" za NVIDIA
        """
        if languages is None:
            languages = ['rs_cyrillic', 'rs_latin']

        self.languages = languages
        self.model_dir = model_storage_directory
        self.use_gpu = use_gpu
        self.gpu_backend = gpu_backend
        self.reader = None
        self._gpu_available = False
        self._actual_backend = "cpu"

        self._initialize_reader()

    def _detect_hardware(self) -> Tuple[bool, str]:
        """
        Automatska detekcija dostupnog hardvera.

        Returns:
            (gpu_available, backend_name)
        """
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            logger.info(f"Dostupni ONNX provideri: {providers}")

            if self.use_gpu:
                if 'DmlExecutionProvider' in providers:
                    logger.info("Koristi se AMD DirectML GPU akceleracija")
                    return True, "directml"
                elif 'CUDAExecutionProvider' in providers:
                    logger.info("Koristi se NVIDIA CUDA GPU akceleracija")
                    return True, "cuda"
        except ImportError:
            logger.warning("onnxruntime nije instaliran, koristim CPU")
        except Exception as e:
            logger.warning(f"Greska pri detekciji GPU: {e}")

        logger.info("Koristi se CPU")
        return False, "cpu"

    def _initialize_reader(self):
        """
        Inicijalizuje EasyOCR reader sa odgovarajucim backend-om.
        """
        try:
            import easyocr
        except ImportError:
            raise ImportError("EasyOCR nije instaliran. Instaliraj sa: pip install easyocr")

        gpu_available, backend = self._detect_hardware()
        self._gpu_available = gpu_available
        self._actual_backend = backend

        try:
            logger.info(f"Inicijalizujem EasyOCR: jezici={self.languages}, GPU={gpu_available}")

            self.reader = easyocr.Reader(
                self.languages,
                gpu=gpu_available,
                model_storage_directory=self.model_dir,
                download_enabled=True,  # Dozvoli download ako modeli ne postoje
                verbose=False
            )

            logger.info(f"OCR Engine inicijalizovan uspesno: "
                       f"jezici={self.languages}, GPU={gpu_available}, backend={backend}")

        except Exception as e:
            logger.error(f"Greska pri inicijalizaciji OCR sa GPU: {e}")
            logger.info("Pokusavam fallback na CPU...")

            try:
                self.reader = easyocr.Reader(
                    self.languages,
                    gpu=False,
                    model_storage_directory=self.model_dir,
                    download_enabled=True,
                    verbose=False
                )
                self._gpu_available = False
                self._actual_backend = "cpu"
                logger.info("OCR Engine inicijalizovan na CPU")
            except Exception as e2:
                logger.error(f"Fallback na CPU nije uspeo: {e2}")
                raise RuntimeError(f"Ne mogu inicijalizovati OCR engine: {e2}")

    def recognize(self, image: np.ndarray,
                  detail: int = 1,
                  paragraph: bool = False,
                  min_size: int = 10,
                  text_threshold: float = 0.7,
                  low_text: float = 0.4) -> List[Dict]:
        """
        Izvrsava OCR na datoj slici.

        Args:
            image: NumPy array slike (grayscale ili BGR)
            detail: 0=samo tekst, 1=tekst+bbox+confidence
            paragraph: Da li grupisati u paragrafe
            min_size: Minimalna velicina teksta
            text_threshold: Prag za detekciju teksta
            low_text: Nizak prag za detekciju

        Returns:
            Lista recnika sa kljucevima: text, confidence, bbox
        """
        if self.reader is None:
            raise RuntimeError("OCR Engine nije inicijalizovan")

        try:
            results = self.reader.readtext(
                image,
                detail=detail,
                paragraph=paragraph,
                min_size=min_size,
                text_threshold=text_threshold,
                low_text=low_text
            )

            parsed_results = []
            for result in results:
                if detail == 1:
                    bbox, text, confidence = result
                    parsed_results.append({
                        'text': text,
                        'confidence': float(confidence),
                        'bbox': bbox
                    })
                else:
                    parsed_results.append({
                        'text': result,
                        'confidence': None,
                        'bbox': None
                    })

            logger.debug(f"OCR prepoznao {len(parsed_results)} rezultata")
            return parsed_results

        except Exception as e:
            logger.error(f"Greska pri OCR prepoznavanju: {e}")
            return []

    def recognize_single_field(self, image: np.ndarray,
                               allowlist: str = None) -> Tuple[str, float]:
        """
        OCR za jedno polje - vraca spojeni tekst i prosecan confidence.

        Args:
            image: Slika polja
            allowlist: Dozvoljeni karakteri (npr. "0123456789" za cifre)

        Returns:
            (tekst, confidence)
        """
        try:
            if allowlist:
                results = self.reader.readtext(
                    image,
                    detail=1,
                    paragraph=False,
                    allowlist=allowlist
                )
            else:
                results = self.reader.readtext(
                    image,
                    detail=1,
                    paragraph=False
                )

            if not results:
                return "", 0.0

            texts = []
            confidences = []

            for bbox, text, confidence in results:
                texts.append(text)
                confidences.append(confidence)

            combined_text = " ".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            return combined_text.strip(), float(avg_confidence)

        except Exception as e:
            logger.error(f"Greska u recognize_single_field: {e}")
            return "", 0.0

    def recognize_digits_only(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Specijalizovano prepoznavanje samo cifara (za JMBG, godine, itd.).

        Args:
            image: Slika polja sa ciframa

        Returns:
            (cifre, confidence)
        """
        # Koristi allowlist za cifre
        text, confidence = self.recognize_single_field(
            image,
            allowlist='0123456789'
        )

        # Dodatno filtriranje - ukloni sve sto nije cifra
        digits_only = ''.join(filter(str.isdigit, text))

        return digits_only, confidence

    def recognize_with_alternatives(self, image: np.ndarray,
                                    num_alternatives: int = 3) -> List[Dict]:
        """
        Vraca vise alternativnih interpretacija teksta.
        Korisno za nesiguran rukopis.

        Args:
            image: Slika polja
            num_alternatives: Broj alternativa

        Returns:
            Lista alternativnih rezultata sortiranih po confidence
        """
        # EasyOCR ne podrzava multiple alternatives direktno,
        # ali mozemo koristiti razlicite pragove
        alternatives = []

        thresholds = [0.7, 0.5, 0.3]
        for threshold in thresholds[:num_alternatives]:
            results = self.recognize(
                image,
                detail=1,
                text_threshold=threshold
            )
            if results:
                combined_text = " ".join([r['text'] for r in results])
                avg_conf = sum([r['confidence'] for r in results]) / len(results)
                alternatives.append({
                    'text': combined_text,
                    'confidence': avg_conf,
                    'threshold': threshold
                })

        # Sortiraj po confidence
        alternatives.sort(key=lambda x: x['confidence'], reverse=True)

        # Ukloni duplikate
        seen = set()
        unique = []
        for alt in alternatives:
            if alt['text'] not in seen:
                seen.add(alt['text'])
                unique.append(alt)

        return unique

    @property
    def is_gpu_enabled(self) -> bool:
        """Da li je GPU aktivan."""
        return self._gpu_available

    @property
    def backend(self) -> str:
        """Trenutni backend (cpu, cuda, directml)."""
        return self._actual_backend

    def get_info(self) -> Dict:
        """Vraca informacije o OCR engine-u."""
        return {
            'languages': self.languages,
            'gpu_enabled': self._gpu_available,
            'backend': self._actual_backend,
            'model_directory': str(self.model_dir) if self.model_dir else None,
            'initialized': self.reader is not None
        }


class OCREngineFactory:
    """Factory za kreiranje OCR engine instance (Singleton pattern)."""

    _instance: Optional[OCREngine] = None

    @classmethod
    def get_instance(cls, **kwargs) -> OCREngine:
        """
        Vraca singleton instancu OCR engine-a.

        Args:
            **kwargs: Argumenti za OCREngine konstruktor (ignorisu se ako vec postoji instanca)
        """
        if cls._instance is None:
            cls._instance = OCREngine(**kwargs)
        return cls._instance

    @classmethod
    def reset(cls):
        """Resetuje singleton instancu (za testiranje)."""
        cls._instance = None

    @classmethod
    def is_initialized(cls) -> bool:
        """Proverava da li je engine inicijalizovan."""
        return cls._instance is not None


def test_ocr_engine():
    """Test funkcija za OCR engine."""
    import cv2

    print("Testiram OCR Engine...")

    engine = OCREngine(
        languages=['rs_cyrillic', 'rs_latin'],
        use_gpu=True
    )

    print(f"Engine info: {engine.get_info()}")

    # Kreiraj test sliku sa tekstom
    test_img = np.ones((100, 400), dtype=np.uint8) * 255
    cv2.putText(test_img, "Test 123", (50, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)

    text, confidence = engine.recognize_single_field(test_img)
    print(f"Prepoznat tekst: '{text}' (confidence: {confidence:.2f})")

    digits, conf = engine.recognize_digits_only(test_img)
    print(f"Prepoznate cifre: '{digits}' (confidence: {conf:.2f})")

    print("Test zavrsen!")


if __name__ == "__main__":
    test_ocr_engine()
