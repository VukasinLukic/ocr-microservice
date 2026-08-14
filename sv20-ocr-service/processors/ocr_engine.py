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

    Drzi DVA odvojena EasyOCR readera (cirilica i latinica) jer ih EasyOCR ne
    dozvoljava u istom Reader-u ("Cyrillic is only compatible with English"
    greska - modeli za cirilicu i latinicu koriste nekompatibilne skupove
    karaktera). Za tekstualna polja se OBA readera pokrecu nad istim ROI-jem
    i zadrzava se rezultat sa visim confidence-om (vidi recognize_single_field) -
    prethodno je servis koristio SAMO cirilicni skup jezika i latinica se
    nikad nije stvarno citala, iako je config/README tvrdio suprotno.
    """

    # Cirilicni EasyOCR jezicki skup - "Fix for Cyrillic is only compatible
    # with English" - EasyOCR zahteva bas ovaj skup pratecih jezika da bi
    # ucitao cirilicni model (ne moze se prosto dodati rs_latin ovde).
    LANGUAGES_CYRILLIC = ["ru", "rs_cyrillic", "be", "bg", "uk", "mn", "en"]
    # Latinicni jezicki skup - rs_latin + en (kompatibilni skup karaktera).
    LANGUAGES_LATIN = ["rs_latin", "en"]

    def __init__(self,
                 languages: List[str] = None,
                 model_storage_directory: str = None,
                 use_gpu: bool = True,
                 gpu_backend: str = "directml"):
        """
        Inicijalizacija OCR engine-a. Uvek ucitava OBA jezicka skupa
        (cirilica + latinica) - `languages` parametar se cuva samo
        informativno (npr. za logove), ne utice na to koji se modeli
        ucitavaju, jer EasyOCR-ova ogranicenja kompatibilnosti jezika ionako
        diktiraju tacno ova dva skupa.

        Args:
            languages: Informativna lista (zadrzano za kompatibilnost API-ja)
            model_storage_directory: Putanja do offline modela
            use_gpu: Da li koristiti GPU
            gpu_backend: "directml" za AMD, "cuda" za NVIDIA
        """
        self.languages = languages or (self.LANGUAGES_CYRILLIC + ["rs_latin"])
        self.model_dir = model_storage_directory
        self.use_gpu = use_gpu
        self.gpu_backend = gpu_backend
        self.reader_cyrillic = None
        self.reader_latin = None
        self._gpu_available = False
        self._actual_backend = "cpu"

        self._initialize_readers()

    def _detect_hardware(self) -> Tuple[bool, str]:
        """
        Automatska detekcija hardvera koji EasyOCR STVARNO koristi.

        VAŽNO (ANALIZA-OCR-SERVISA.md 4.3): EasyOCR interno koristi PyTorch,
        NE onnxruntime - proveravanje `onnxruntime` DirectML providera ovde
        je ranije davalo potpuno pogrešnu sliku (servis je "video" AMD GPU
        preko onnxruntime-a, ali je EasyOCR i dalje tiho radio na CPU-u jer
        `torch.cuda.is_available()` na AMD/Windows-u bez ROCm-a vraća False).
        Sad se proverava ono što EasyOCR stvarno koristi.

        AMD GPU akceleracija bi zahtevala `torch-directml` paket + eksplicitno
        stavljanje tenzora na `dml` device u pozivnom kodu (EasyOCR ne
        podržava proizvoljan torch device "od kutije") - nije urađeno u ovom
        krugu jer je rizično/nepotvrđeno, a Microsoft-ov DirectML repo je u
        "maintenance mode" (vidi ANALIZA-OCR-SERVISA.md 4.3 za detalje i
        alternative poput OnnxTR/docTR koje stvarno prolaze kroz onnxruntime).

        Returns:
            (gpu_available, backend_name)
        """
        if not self.use_gpu:
            logger.info("GPU isključen konfiguracijom, koristim CPU")
            return False, "cpu"

        try:
            import torch
            if torch.cuda.is_available():
                logger.info("Koristi se NVIDIA CUDA GPU akceleracija")
                return True, "cuda"
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Greska pri proveri torch.cuda: {e}")

        try:
            import torch_directml  # noqa: F401
            logger.info("Koristi se AMD/Intel DirectML GPU akceleracija (torch_directml)")
            return True, "directml"
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Greska pri proveri torch_directml: {e}")

        logger.info("Nijedan GPU backend nije dostupan za EasyOCR (torch je CPU-only build "
                   "i/ili torch_directml nije instaliran) - koristi se CPU")
        return False, "cpu"

    def _initialize_readers(self):
        """
        Inicijalizuje OBA EasyOCR readera (cirilica i latinica) sa
        odgovarajucim backend-om.
        """
        try:
            import easyocr
        except ImportError:
            raise ImportError("EasyOCR nije instaliran. Instaliraj sa: pip install easyocr")

        gpu_available, backend = self._detect_hardware()
        self._gpu_available = gpu_available
        self._actual_backend = backend

        self.reader_cyrillic = self._build_reader(easyocr, self.LANGUAGES_CYRILLIC, gpu_available)
        self.reader_latin = self._build_reader(easyocr, self.LANGUAGES_LATIN, gpu_available)

        logger.info(f"OCR Engine inicijalizovan uspesno: cirilica={self.LANGUAGES_CYRILLIC}, "
                   f"latinica={self.LANGUAGES_LATIN}, GPU={self._gpu_available}, backend={self._actual_backend}")

    def _build_reader(self, easyocr_module, languages: List[str], gpu_available: bool):
        """
        Gradi jedan EasyOCR Reader za dati jezicki skup, sa fallback-om na
        CPU ako GPU inicijalizacija ne uspe.
        """
        try:
            logger.info(f"Inicijalizujem EasyOCR reader: jezici={languages}, GPU={gpu_available}")
            return easyocr_module.Reader(
                languages,
                gpu=gpu_available,
                model_storage_directory=self.model_dir,
                download_enabled=True,  # Dozvoli download ako modeli ne postoje
                verbose=False
            )
        except Exception as e:
            logger.error(f"Greska pri inicijalizaciji {languages} sa GPU: {e}")
            logger.info("Pokusavam fallback na CPU...")

            try:
                reader = easyocr_module.Reader(
                    languages,
                    gpu=False,
                    model_storage_directory=self.model_dir,
                    download_enabled=True,
                    verbose=False
                )
                self._gpu_available = False
                self._actual_backend = "cpu"
                logger.info(f"Reader {languages} inicijalizovan na CPU")
                return reader
            except Exception as e2:
                logger.error(f"Fallback na CPU nije uspeo za {languages}: {e2}")
                raise RuntimeError(f"Ne mogu inicijalizovati OCR reader {languages}: {e2}")

    def recognize(self, image: np.ndarray,
                  detail: int = 1,
                  paragraph: bool = False,
                  min_size: int = 10,
                  text_threshold: float = 0.7,
                  low_text: float = 0.4) -> List[Dict]:
        """
        Izvrsava OCR na datoj slici preko cirilicnog readera (primarno pismo
        stampanih delova ŠV-20 obrasca). Koristi se uglavnom za "sirovi",
        pregledni OCR celog dokumenta (/api/ocr/process-raw, debug) - za
        pojedinacna polja gde pismo realno varira koristi
        recognize_single_field, koja pokrece OBA readera.

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
        if self.reader_cyrillic is None:
            raise RuntimeError("OCR Engine nije inicijalizovan")

        try:
            results = self.reader_cyrillic.readtext(
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

    def _recognize_with_reader(self, reader, image: np.ndarray,
                                allowlist: str = None) -> Tuple[str, float]:
        """Pokrece jedan konkretan EasyOCR reader nad ROI-jem jednog polja."""
        try:
            if allowlist:
                results = reader.readtext(image, detail=1, paragraph=False, allowlist=allowlist)
            else:
                results = reader.readtext(image, detail=1, paragraph=False)

            if not results:
                return "", 0.0

            texts = [text for _, text, _ in results]
            confidences = [confidence for _, _, confidence in results]

            combined_text = " ".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            return combined_text.strip(), float(avg_confidence)

        except Exception as e:
            logger.error(f"Greska u _recognize_with_reader: {e}")
            return "", 0.0

    def recognize_single_field(self, image: np.ndarray,
                               allowlist: str = None) -> Tuple[str, float]:
        """
        OCR za jedno polje - vraca spojeni tekst i prosecan confidence.

        Pokrece OBA readera (cirilica + latinica) nad istim ROI-jem i vraca
        rezultat sa VISIM prosecnim confidence-om (opcija B iz WS3 -
        vidi ANALIZA-OCR-SERVISA.md 4.2). Izuzetak: kad je allowlist cisto
        numericki (JMBG, godine i sl.), pismo nije relevantno za cifre pa se
        koristi samo cirilicni reader (vec sadrzi 'en') da se ne duplira
        nepotreban trosak vremena.

        Args:
            image: Slika polja
            allowlist: Dozvoljeni karakteri (npr. "0123456789" za cifre)

        Returns:
            (tekst, confidence)
        """
        if allowlist and set(allowlist) <= set('0123456789'):
            return self._recognize_with_reader(self.reader_cyrillic, image, allowlist)

        text_cyrillic, conf_cyrillic = self._recognize_with_reader(self.reader_cyrillic, image, allowlist)
        text_latin, conf_latin = self._recognize_with_reader(self.reader_latin, image, allowlist)

        if conf_latin > conf_cyrillic:
            logger.info(f"[DUAL-SCRIPT] Latinica pobedila: '{text_latin}' (conf={conf_latin:.2f}) "
                       f"> cirilica '{text_cyrillic}' (conf={conf_cyrillic:.2f})")
            return text_latin, conf_latin

        return text_cyrillic, conf_cyrillic

    def recognize_digits_only(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Specijalizovano prepoznavanje samo cifara (za JMBG, godine, itd.).

        Args:
            image: Slika polja sa ciframa

        Returns:
            (cifre, confidence)
        """
        # Koristi strict allowlist za cifre
        # Ovo sprecava da se '|', 'I', 'l' i slicno citaju kao cifre
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
            'languages_cyrillic': self.LANGUAGES_CYRILLIC,
            'languages_latin': self.LANGUAGES_LATIN,
            'languages': self.languages,
            'gpu_enabled': self._gpu_available,
            'backend': self._actual_backend,
            'model_directory': str(self.model_dir) if self.model_dir else None,
            'initialized': self.reader_cyrillic is not None and self.reader_latin is not None
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
