# Arhitektura i Implementacija Offline Python OCR Mikroservisa za SV-20 Obrazac

## 1. Pregled Arhitekture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           JAVA KLIJENT                                   │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────────────┐ │
│  │ SV20Obrazac │───>│ OCRService   │───>│ HTTP MultipartBody Request  │ │
│  │ Controller  │<───│ .java        │<───│ JSON Response               │ │
│  └─────────────┘    └──────────────┘    └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                │ HTTP POST /api/ocr/process
                                │ Content-Type: multipart/form-data
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      PYTHON OCR MIKROSERVIS                              │
│                         (FastAPI + Uvicorn)                              │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                         api/server.py                                ││
│  │  POST /api/ocr/process  │  GET /api/health  │  GET /api/template    ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                │                                         │
│         ┌──────────────────────┼──────────────────────┐                 │
│         ▼                      ▼                      ▼                 │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────────┐         │
│  │ image_      │      │ ocr_engine  │      │ omr_logic.py    │         │
│  │ processor.py│      │ .py         │      │ (Mark Recogn.)  │         │
│  │             │      │             │      │                 │         │
│  │ - Deskew    │      │ - EasyOCR   │      │ - Pixel Density │         │
│  │ - Denoise   │      │ - sr_cyrl   │      │ - Contour Det.  │         │
│  │ - ROI Crop  │      │ - sr_latn   │      │ - Checkbox Det. │         │
│  │ - Line Rem. │      │ - DirectML  │      │                 │         │
│  └─────────────┘      └─────────────┘      └─────────────────┘         │
│         │                      │                      │                 │
│         └──────────────────────┼──────────────────────┘                 │
│                                ▼                                         │
│                    ┌─────────────────────┐                              │
│                    │ validators.py       │                              │
│                    │ - JMBG (13 cifara)  │                              │
│                    │ - Datum formati     │                              │
│                    │ - Indeks format     │                              │
│                    └─────────────────────┘                              │
│                                │                                         │
│                                ▼                                         │
│                    ┌─────────────────────┐                              │
│                    │ JSON Response       │                              │
│                    │ {polja: [...]}      │                              │
│                    └─────────────────────┘                              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Struktura Projekta

```
sv20-ocr-service/
│
├── main.py                      # Entry point
├── config.py                    # Konfiguracija (port, GPU, putanje)
├── requirements.txt             # Python dependencies
├── download_models.py           # Offline priprema modela
│
├── api/
│   ├── __init__.py
│   └── server.py                # FastAPI endpoints
│
├── processors/
│   ├── __init__.py
│   ├── image_processor.py       # OpenCV pipeline
│   ├── ocr_engine.py            # EasyOCR wrapper
│   ├── omr_logic.py             # Optical Mark Recognition
│   └── validators.py            # Format validacija
│
├── templates/
│   └── sv20_template.json       # Koordinate polja
│
├── models/                      # EasyOCR modeli (offline)
│   ├── sr_cyrl/
│   └── sr_latn/
│
├── tests/
│   ├── test_ocr.py
│   ├── test_omr.py
│   └── sample_images/
│
└── logs/
    └── ocr_service.log
```

---

## 3. Detaljna Specifikacija Modula

### 3.1 config.py - Konfiguracija

```python
import os
from pathlib import Path

class Config:
    # Server
    HOST = "127.0.0.1"
    PORT = 9001

    # Paths
    BASE_DIR = Path(__file__).parent
    MODELS_DIR = BASE_DIR / "models"
    TEMPLATES_DIR = BASE_DIR / "templates"
    LOGS_DIR = BASE_DIR / "logs"

    # OCR Settings
    OCR_LANGUAGES = ['sr_cyrl', 'sr_latn']
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

    @classmethod
    def detect_gpu(cls):
        """Automatska detekcija GPU-a"""
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if 'DmlExecutionProvider' in providers:
                return "directml"
            elif 'CUDAExecutionProvider' in providers:
                return "cuda"
        except:
            pass
        return "cpu"
```

### 3.2 templates/sv20_template.json - Koordinatni Sistem

```json
{
  "document": {
    "name": "SV-20 Obrazac",
    "version": "1.0",
    "pages": 1,
    "expected_dpi": 300
  },
  "fields": [
    {
      "id": 1,
      "name": "ime_studenta",
      "type": "TEXT",
      "coordinates": {
        "x": 150,
        "y": 320,
        "width": 400,
        "height": 50
      },
      "page": 1,
      "required": true,
      "preprocessing": ["denoise", "deskew"],
      "postprocessing": ["capitalize_first"]
    },
    {
      "id": 2,
      "name": "prezime_studenta",
      "type": "TEXT",
      "coordinates": {
        "x": 150,
        "y": 380,
        "width": 400,
        "height": 50
      },
      "page": 1,
      "required": true,
      "preprocessing": ["denoise", "deskew"],
      "postprocessing": ["capitalize_first"]
    },
    {
      "id": 3,
      "name": "jmbg",
      "type": "NUMERIC",
      "coordinates": {
        "x": 150,
        "y": 450,
        "width": 520,
        "height": 60
      },
      "page": 1,
      "required": true,
      "preprocessing": ["remove_comb_boxes", "denoise"],
      "validation": {
        "length": 13,
        "pattern": "^[0-9]{13}$"
      },
      "comb_box": {
        "enabled": true,
        "cell_count": 13,
        "cell_width": 40
      }
    },
    {
      "id": 4,
      "name": "broj_indeksa",
      "type": "ALPHANUMERIC",
      "coordinates": {
        "x": 150,
        "y": 520,
        "width": 300,
        "height": 50
      },
      "page": 1,
      "required": true,
      "preprocessing": ["denoise"],
      "validation": {
        "pattern": "^\\d{4}/\\d{4}$"
      }
    },
    {
      "id": 5,
      "name": "pol",
      "type": "OMR_SINGLE",
      "coordinates": {
        "x": 600,
        "y": 320,
        "width": 200,
        "height": 60
      },
      "page": 1,
      "required": true,
      "omr_options": [
        {"label": "M", "x": 620, "y": 330, "width": 40, "height": 40},
        {"label": "Z", "x": 720, "y": 330, "width": 40, "height": 40}
      ]
    },
    {
      "id": 6,
      "name": "datum_rodjenja",
      "type": "DATE",
      "coordinates": {
        "x": 150,
        "y": 590,
        "width": 250,
        "height": 50
      },
      "page": 1,
      "required": true,
      "preprocessing": ["denoise", "remove_lines"],
      "validation": {
        "format": "DD.MM.YYYY"
      }
    },
    {
      "id": 7,
      "name": "mesto_rodjenja",
      "type": "TEXT",
      "coordinates": {
        "x": 450,
        "y": 590,
        "width": 350,
        "height": 50
      },
      "page": 1,
      "required": true,
      "preprocessing": ["denoise", "deskew"]
    },
    {
      "id": 8,
      "name": "adresa_stanovanja",
      "type": "TEXT",
      "coordinates": {
        "x": 150,
        "y": 660,
        "width": 650,
        "height": 50
      },
      "page": 1,
      "required": true,
      "preprocessing": ["denoise", "remove_lines"]
    },
    {
      "id": 9,
      "name": "studijski_program",
      "type": "TEXT",
      "coordinates": {
        "x": 150,
        "y": 730,
        "width": 500,
        "height": 50
      },
      "page": 1,
      "required": true,
      "preprocessing": ["denoise"]
    },
    {
      "id": 10,
      "name": "godina_studija",
      "type": "OMR_SINGLE",
      "coordinates": {
        "x": 150,
        "y": 800,
        "width": 400,
        "height": 50
      },
      "page": 1,
      "required": true,
      "omr_options": [
        {"label": "1", "x": 160, "y": 810, "width": 30, "height": 30},
        {"label": "2", "x": 220, "y": 810, "width": 30, "height": 30},
        {"label": "3", "x": 280, "y": 810, "width": 30, "height": 30},
        {"label": "4", "x": 340, "y": 810, "width": 30, "height": 30}
      ]
    },
    {
      "id": 11,
      "name": "skolska_godina",
      "type": "ALPHANUMERIC",
      "coordinates": {
        "x": 600,
        "y": 800,
        "width": 200,
        "height": 50
      },
      "page": 1,
      "required": true,
      "preprocessing": ["denoise"],
      "validation": {
        "pattern": "^\\d{4}/\\d{4}$"
      }
    },
    {
      "id": 12,
      "name": "semestar",
      "type": "OMR_SINGLE",
      "coordinates": {
        "x": 150,
        "y": 870,
        "width": 300,
        "height": 50
      },
      "page": 1,
      "required": true,
      "omr_options": [
        {"label": "Zimski", "x": 160, "y": 880, "width": 80, "height": 30},
        {"label": "Letnji", "x": 280, "y": 880, "width": 80, "height": 30}
      ]
    }
  ]
}
```

### 3.3 processors/image_processor.py - OpenCV Pipeline

```python
import cv2
import numpy as np
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    """
    OpenCV pipeline za obradu slike SV-20 obrasca.
    Podržava: deskew, denoise, ROI cropping, uklanjanje linija/kućica.
    """

    def __init__(self, target_dpi: int = 300):
        self.target_dpi = target_dpi

    def load_image(self, image_path: str) -> np.ndarray:
        """Učitava sliku u grayscale formatu."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Ne mogu učitati sliku: {image_path}")
        return img

    def to_grayscale(self, img: np.ndarray) -> np.ndarray:
        """Konvertuje u grayscale ako nije već."""
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def deskew(self, img: np.ndarray) -> np.ndarray:
        """
        Ispravlja nagib slike koristeći Hough Line Transform.
        """
        gray = self.to_grayscale(img)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100,
                                 minLineLength=100, maxLineGap=10)

        if lines is None:
            return img

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 45:
                angles.append(angle)

        if not angles:
            return img

        median_angle = np.median(angles)

        if abs(median_angle) < 0.5:
            return img

        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)

        logger.info(f"Deskew: korigovano {median_angle:.2f} stepeni")
        return rotated

    def denoise(self, img: np.ndarray) -> np.ndarray:
        """
        Uklanja šum koristeći Non-local Means Denoising.
        """
        gray = self.to_grayscale(img)
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10,
                                            templateWindowSize=7,
                                            searchWindowSize=21)
        return denoised

    def binarize(self, img: np.ndarray,
                 method: str = "adaptive") -> np.ndarray:
        """
        Binarizacija slike - priprema za OCR.

        Args:
            method: "adaptive" ili "otsu"
        """
        gray = self.to_grayscale(img)

        if method == "adaptive":
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
        else:
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

        return binary

    def crop_roi(self, img: np.ndarray,
                 x: int, y: int, width: int, height: int) -> np.ndarray:
        """
        Iseca Region of Interest (ROI) na osnovu koordinata iz template-a.
        """
        h, w = img.shape[:2]

        x = max(0, min(x, w))
        y = max(0, min(y, h))
        x2 = max(0, min(x + width, w))
        y2 = max(0, min(y + height, h))

        roi = img[y:y2, x:x2]

        if roi.size == 0:
            raise ValueError(f"Prazan ROI: ({x},{y}) - ({x2},{y2})")

        return roi

    def remove_horizontal_lines(self, img: np.ndarray,
                                 min_length: int = 100) -> np.ndarray:
        """
        Uklanja horizontalne linije (npr. linije za pisanje).
        Koristi morfološke operacije.
        """
        gray = self.to_grayscale(img)
        binary = cv2.adaptiveThreshold(gray, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)

        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (min_length, 1)
        )
        detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                                          horizontal_kernel, iterations=2)

        repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        result = cv2.subtract(binary, detected_lines)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE,
                                  repair_kernel, iterations=1)

        result = cv2.bitwise_not(result)

        logger.info("Uklonjene horizontalne linije")
        return result

    def remove_comb_boxes(self, img: np.ndarray,
                          cell_count: int = 13,
                          cell_width: int = 40) -> np.ndarray:
        """
        Uklanja vertikalne linije kućica (comb-boxes) kod JMBG polja.
        Ostavlja samo rukopis.
        """
        gray = self.to_grayscale(img)
        binary = cv2.adaptiveThreshold(gray, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)

        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, img.shape[0] // 3)
        )
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                                          vertical_kernel, iterations=2)

        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (img.shape[1] // 2, 1)
        )
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                                            horizontal_kernel, iterations=2)

        lines = cv2.add(vertical_lines, horizontal_lines)

        result = cv2.subtract(binary, lines)

        cleanup_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE,
                                  cleanup_kernel, iterations=1)

        result = cv2.bitwise_not(result)

        logger.info(f"Uklonjene kućice ({cell_count} ćelija)")
        return result

    def enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """
        Pojačava kontrast koristeći CLAHE.
        """
        gray = self.to_grayscale(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return enhanced

    def process_field(self, img: np.ndarray,
                      field_config: dict) -> np.ndarray:
        """
        Primenjuje odgovarajuće preprocessing korake za dato polje.
        """
        coords = field_config['coordinates']
        roi = self.crop_roi(img, coords['x'], coords['y'],
                           coords['width'], coords['height'])

        preprocessing = field_config.get('preprocessing', [])

        for step in preprocessing:
            if step == 'deskew':
                roi = self.deskew(roi)
            elif step == 'denoise':
                roi = self.denoise(roi)
            elif step == 'remove_lines':
                roi = self.remove_horizontal_lines(roi)
            elif step == 'remove_comb_boxes':
                comb_config = field_config.get('comb_box', {})
                roi = self.remove_comb_boxes(
                    roi,
                    cell_count=comb_config.get('cell_count', 13),
                    cell_width=comb_config.get('cell_width', 40)
                )
            elif step == 'enhance_contrast':
                roi = self.enhance_contrast(roi)

        roi = self.binarize(roi)

        return roi
```

### 3.4 processors/ocr_engine.py - EasyOCR sa DirectML

```python
import easyocr
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging
import onnxruntime as ort
from pathlib import Path

logger = logging.getLogger(__name__)

class OCREngine:
    """
    EasyOCR wrapper sa podrškom za AMD DirectML i offline rad.
    Učitava srpsku ćirilicu i latinicu istovremeno.
    """

    def __init__(self,
                 languages: List[str] = ['sr_cyrl', 'sr_latn'],
                 model_storage_directory: str = None,
                 use_gpu: bool = True,
                 gpu_backend: str = "directml"):
        """
        Inicijalizacija OCR engine-a.

        Args:
            languages: Lista jezika za prepoznavanje
            model_storage_directory: Putanja do offline modela
            use_gpu: Da li koristiti GPU
            gpu_backend: "directml" za AMD, "cuda" za NVIDIA
        """
        self.languages = languages
        self.model_dir = model_storage_directory
        self.use_gpu = use_gpu
        self.gpu_backend = gpu_backend
        self.reader = None

        self._initialize_reader()

    def _detect_hardware(self) -> Tuple[bool, str]:
        """
        Automatska detekcija dostupnog hardvera.
        """
        providers = ort.get_available_providers()
        logger.info(f"Dostupni ONNX provideri: {providers}")

        if self.use_gpu:
            if 'DmlExecutionProvider' in providers:
                logger.info("Koristi se AMD DirectML GPU akceleracija")
                return True, "directml"
            elif 'CUDAExecutionProvider' in providers:
                logger.info("Koristi se NVIDIA CUDA GPU akceleracija")
                return True, "cuda"

        logger.info("Koristi se CPU")
        return False, "cpu"

    def _initialize_reader(self):
        """
        Inicijalizuje EasyOCR reader sa odgovarajućim backend-om.
        """
        gpu_available, backend = self._detect_hardware()

        try:
            self.reader = easyocr.Reader(
                self.languages,
                gpu=gpu_available,
                model_storage_directory=self.model_dir,
                download_enabled=False,
                verbose=False
            )
            logger.info(f"OCR Engine inicijalizovan: jezici={self.languages}, "
                       f"GPU={gpu_available}, backend={backend}")
        except Exception as e:
            logger.error(f"Greška pri inicijalizaciji OCR: {e}")
            logger.info("Pokušavam fallback na CPU...")
            self.reader = easyocr.Reader(
                self.languages,
                gpu=False,
                model_storage_directory=self.model_dir,
                download_enabled=False,
                verbose=False
            )

    def recognize(self, image: np.ndarray,
                  detail: int = 1,
                  paragraph: bool = False) -> List[Dict]:
        """
        Izvršava OCR na datoj slici.

        Args:
            image: NumPy array slike (grayscale ili BGR)
            detail: 0=samo tekst, 1=tekst+bbox+confidence
            paragraph: Da li grupisati u paragrafe

        Returns:
            Lista rečnika sa ključevima: text, confidence, bbox
        """
        if self.reader is None:
            raise RuntimeError("OCR Engine nije inicijalizovan")

        try:
            results = self.reader.readtext(
                image,
                detail=detail,
                paragraph=paragraph
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

            return parsed_results

        except Exception as e:
            logger.error(f"Greška pri OCR prepoznavanju: {e}")
            return []

    def recognize_single_field(self, image: np.ndarray) -> Tuple[str, float]:
        """
        OCR za jedno polje - vraća spojeni tekst i prosečan confidence.
        """
        results = self.recognize(image, detail=1)

        if not results:
            return "", 0.0

        texts = [r['text'] for r in results]
        confidences = [r['confidence'] for r in results]

        combined_text = " ".join(texts)
        avg_confidence = sum(confidences) / len(confidences)

        return combined_text.strip(), avg_confidence

    def recognize_digits_only(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Specijalizovano prepoznavanje samo cifara (za JMBG).
        """
        results = self.recognize(image, detail=1)

        if not results:
            return "", 0.0

        digits = []
        confidences = []

        for r in results:
            for char in r['text']:
                if char.isdigit():
                    digits.append(char)
                    confidences.append(r['confidence'])

        if not digits:
            return "", 0.0

        return "".join(digits), sum(confidences) / len(confidences)


class OCREngineFactory:
    """Factory za kreiranje OCR engine instance."""

    _instance: Optional[OCREngine] = None

    @classmethod
    def get_instance(cls, **kwargs) -> OCREngine:
        """Singleton pattern - jedna instanca OCR engine-a."""
        if cls._instance is None:
            cls._instance = OCREngine(**kwargs)
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset instance (za testiranje)."""
        cls._instance = None
```

### 3.5 processors/omr_logic.py - Optical Mark Recognition

```python
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class OMRProcessor:
    """
    Optical Mark Recognition - detekcija zaokruženih/označenih opcija.
    Koristi se za polja kao što su: Pol, Godina studija, Semestar.
    """

    def __init__(self,
                 fill_threshold: float = 0.3,
                 min_contour_area: int = 100):
        """
        Args:
            fill_threshold: Procenat popunjenosti za detekciju (0.0-1.0)
            min_contour_area: Minimalna površina konture
        """
        self.fill_threshold = fill_threshold
        self.min_contour_area = min_contour_area

    def detect_marked_option(self,
                             image: np.ndarray,
                             options: List[Dict]) -> Dict:
        """
        Detektuje koja opcija je označena/zaokružena.

        Args:
            image: Slika polja sa opcijama
            options: Lista opcija sa koordinatama iz template-a
                    [{"label": "M", "x": 10, "y": 10, "width": 40, "height": 40}, ...]

        Returns:
            Dict sa detektovanom opcijom i confidence
        """
        gray = self._to_grayscale(image)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        results = []

        for option in options:
            roi = self._crop_option_roi(binary, option)
            fill_ratio = self._calculate_fill_ratio(roi)

            results.append({
                'label': option['label'],
                'fill_ratio': fill_ratio,
                'detected': fill_ratio >= self.fill_threshold
            })

            logger.debug(f"Opcija {option['label']}: fill_ratio={fill_ratio:.3f}")

        detected_options = [r for r in results if r['detected']]

        if len(detected_options) == 1:
            return {
                'value': detected_options[0]['label'],
                'confidence': min(detected_options[0]['fill_ratio'] / self.fill_threshold, 1.0),
                'all_options': results
            }
        elif len(detected_options) > 1:
            best = max(detected_options, key=lambda x: x['fill_ratio'])
            return {
                'value': best['label'],
                'confidence': 0.7,
                'warning': 'Više opcija detektovano',
                'all_options': results
            }
        else:
            return {
                'value': None,
                'confidence': 0.0,
                'warning': 'Nijedna opcija nije detektovana',
                'all_options': results
            }

    def detect_checkbox(self,
                        image: np.ndarray,
                        checkbox_coords: Dict) -> Tuple[bool, float]:
        """
        Detektuje da li je checkbox označen.

        Returns:
            (is_checked, confidence)
        """
        gray = self._to_grayscale(image)
        roi = self._crop_option_roi(gray, checkbox_coords)

        binary = cv2.adaptiveThreshold(
            roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        has_mark = False
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h) if h > 0 else 0
                if 0.3 < aspect_ratio < 3.0:
                    has_mark = True
                    break

        fill_ratio = self._calculate_fill_ratio(binary)

        is_checked = has_mark or fill_ratio >= self.fill_threshold
        confidence = fill_ratio if is_checked else 1.0 - fill_ratio

        return is_checked, confidence

    def detect_circled_text(self,
                            image: np.ndarray,
                            options: List[Dict]) -> Dict:
        """
        Detektuje tekst koji je zaokružen olovkom.
        Koristi detekciju kontura elipsi/krugova.
        """
        gray = self._to_grayscale(image)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        circles = []
        for contour in contours:
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                (cx, cy), (ma, MA), angle = ellipse

                if ma > 0:
                    ratio = MA / ma
                    if 0.5 < ratio < 2.0:
                        circles.append({
                            'center': (int(cx), int(cy)),
                            'axes': (int(ma/2), int(MA/2)),
                            'contour': contour
                        })

        for option in options:
            opt_center = (
                option['x'] + option['width'] // 2,
                option['y'] + option['height'] // 2
            )

            for circle in circles:
                dist = np.sqrt(
                    (circle['center'][0] - opt_center[0])**2 +
                    (circle['center'][1] - opt_center[1])**2
                )

                if dist < max(option['width'], option['height']):
                    return {
                        'value': option['label'],
                        'confidence': 0.85,
                        'method': 'circle_detection'
                    }

        return self.detect_marked_option(image, options)

    def _to_grayscale(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _crop_option_roi(self, image: np.ndarray, coords: Dict) -> np.ndarray:
        x = coords.get('x', 0)
        y = coords.get('y', 0)
        w = coords.get('width', image.shape[1])
        h = coords.get('height', image.shape[0])

        h_img, w_img = image.shape[:2]
        x = max(0, min(x, w_img))
        y = max(0, min(y, h_img))
        x2 = max(0, min(x + w, w_img))
        y2 = max(0, min(y + h, h_img))

        return image[y:y2, x:x2]

    def _calculate_fill_ratio(self, binary_image: np.ndarray) -> float:
        """Računa procenat belih piksela u binarnoj slici."""
        if binary_image.size == 0:
            return 0.0

        white_pixels = cv2.countNonZero(binary_image)
        total_pixels = binary_image.shape[0] * binary_image.shape[1]

        return white_pixels / total_pixels if total_pixels > 0 else 0.0
```

### 3.6 processors/validators.py - Validacija Formata

```python
import re
from datetime import datetime
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)

class FieldValidator:
    """
    Validacija ekstraktovanih OCR vrednosti prema pravilima dokumentacije.
    """

    JMBG_LENGTH = 13
    INDEX_PATTERN = r"^\d{4}/\d{4}$"
    DATE_PATTERNS = [
        (r"^\d{2}\.\d{2}\.\d{4}$", "%d.%m.%Y"),
        (r"^\d{2}/\d{2}/\d{4}$", "%d/%m/%Y"),
        (r"^\d{4}-\d{2}-\d{2}$", "%Y-%m-%d"),
    ]

    @classmethod
    def validate_jmbg(cls, value: str) -> Tuple[bool, str, Optional[str]]:
        """
        Validira JMBG (Jedinstveni matični broj građana).

        Format: DDMMGGGRRBBBK
        - DD: dan rođenja
        - MM: mesec rođenja
        - GGG: poslednje 3 cifre godine
        - RR: region
        - BBB: jedinstveni broj
        - K: kontrolna cifra

        Returns:
            (is_valid, cleaned_value, error_message)
        """
        cleaned = re.sub(r'\D', '', value)

        if len(cleaned) != cls.JMBG_LENGTH:
            return False, cleaned, f"JMBG mora imati {cls.JMBG_LENGTH} cifara, ima {len(cleaned)}"

        if not cleaned.isdigit():
            return False, cleaned, "JMBG mora sadržati samo cifre"

        if cls._validate_jmbg_checksum(cleaned):
            return True, cleaned, None
        else:
            return False, cleaned, "Neispravna kontrolna cifra JMBG-a"

    @staticmethod
    def _validate_jmbg_checksum(jmbg: str) -> bool:
        """
        Validira kontrolnu cifru JMBG-a po modulu 11.
        """
        if len(jmbg) != 13:
            return False

        weights = [7, 6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 2]

        total = sum(int(jmbg[i]) * weights[i] for i in range(12))
        remainder = total % 11

        if remainder == 0:
            expected_check = 0
        elif remainder == 1:
            return False
        else:
            expected_check = 11 - remainder

        return int(jmbg[12]) == expected_check

    @classmethod
    def validate_index(cls, value: str) -> Tuple[bool, str, Optional[str]]:
        """
        Validira broj indeksa studenta.
        Format: GGGG/BBBB (npr. 2023/0342)
        """
        cleaned = value.strip()

        cleaned = re.sub(r'\s+', '', cleaned)
        cleaned = cleaned.replace('l', '1').replace('O', '0').replace('o', '0')

        if re.match(cls.INDEX_PATTERN, cleaned):
            return True, cleaned, None

        digits = re.sub(r'\D', '', cleaned)
        if len(digits) == 8:
            formatted = f"{digits[:4]}/{digits[4:]}"
            return True, formatted, None

        return False, cleaned, "Indeks mora biti u formatu GGGG/BBBB"

    @classmethod
    def validate_date(cls, value: str) -> Tuple[bool, str, Optional[str]]:
        """
        Validira datum i vraća u standardnom formatu DD.MM.YYYY.
        """
        cleaned = value.strip()

        for pattern, date_format in cls.DATE_PATTERNS:
            if re.match(pattern, cleaned):
                try:
                    date_obj = datetime.strptime(cleaned, date_format)

                    if date_obj.year < 1900 or date_obj.year > datetime.now().year:
                        return False, cleaned, f"Nevažeća godina: {date_obj.year}"

                    formatted = date_obj.strftime("%d.%m.%Y")
                    return True, formatted, None
                except ValueError as e:
                    return False, cleaned, f"Nevažeći datum: {e}"

        digits = re.sub(r'\D', '', cleaned)
        if len(digits) == 8:
            try:
                formatted = f"{digits[:2]}.{digits[2:4]}.{digits[4:]}"
                date_obj = datetime.strptime(formatted, "%d.%m.%Y")
                return True, formatted, None
            except ValueError:
                pass

        return False, cleaned, "Datum mora biti u formatu DD.MM.YYYY"

    @classmethod
    def validate_text(cls, value: str,
                      min_length: int = 1,
                      max_length: int = 500,
                      required: bool = True) -> Tuple[bool, str, Optional[str]]:
        """
        Validira tekstualno polje.
        """
        cleaned = value.strip()

        cleaned = re.sub(r'\s+', ' ', cleaned)

        if required and not cleaned:
            return False, cleaned, "Polje je obavezno"

        if len(cleaned) < min_length:
            return False, cleaned, f"Minimalna dužina je {min_length} karaktera"

        if len(cleaned) > max_length:
            return False, cleaned[:max_length], f"Tekst skraćen na {max_length} karaktera"

        return True, cleaned, None

    @classmethod
    def validate_numeric(cls, value: str,
                         min_val: Optional[int] = None,
                         max_val: Optional[int] = None) -> Tuple[bool, str, Optional[str]]:
        """
        Validira numeričko polje.
        """
        cleaned = re.sub(r'\D', '', value)

        if not cleaned:
            return False, "", "Očekuje se numerička vrednost"

        num_value = int(cleaned)

        if min_val is not None and num_value < min_val:
            return False, cleaned, f"Vrednost mora biti >= {min_val}"

        if max_val is not None and num_value > max_val:
            return False, cleaned, f"Vrednost mora biti <= {max_val}"

        return True, cleaned, None

    @classmethod
    def validate_field(cls, value: str, field_config: dict) -> dict:
        """
        Validira polje na osnovu konfiguracije iz template-a.
        """
        field_type = field_config.get('type', 'TEXT')
        required = field_config.get('required', False)
        validation = field_config.get('validation', {})

        if field_type == 'TEXT':
            is_valid, cleaned, error = cls.validate_text(
                value, required=required,
                min_length=validation.get('min_length', 1),
                max_length=validation.get('max_length', 500)
            )

        elif field_type == 'NUMERIC':
            if validation.get('length') == 13:
                is_valid, cleaned, error = cls.validate_jmbg(value)
            else:
                is_valid, cleaned, error = cls.validate_numeric(
                    value,
                    min_val=validation.get('min'),
                    max_val=validation.get('max')
                )

        elif field_type == 'ALPHANUMERIC':
            pattern = validation.get('pattern')
            if pattern and 'indeks' in field_config.get('name', '').lower():
                is_valid, cleaned, error = cls.validate_index(value)
            else:
                is_valid, cleaned, error = cls.validate_text(value, required=required)

        elif field_type == 'DATE':
            is_valid, cleaned, error = cls.validate_date(value)

        else:
            is_valid, cleaned, error = True, value.strip(), None

        return {
            'is_valid': is_valid,
            'original_value': value,
            'cleaned_value': cleaned,
            'error': error
        }


def postprocess_text(text: str, operations: List[str]) -> str:
    """
    Primenjuje post-processing operacije na OCR tekst.
    """
    result = text

    for op in operations:
        if op == 'capitalize_first':
            result = result.capitalize()
        elif op == 'uppercase':
            result = result.upper()
        elif op == 'lowercase':
            result = result.lower()
        elif op == 'strip_spaces':
            result = result.strip()
        elif op == 'normalize_spaces':
            result = re.sub(r'\s+', ' ', result).strip()

    return result
```

### 3.7 api/server.py - FastAPI Endpoints

```python
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import json
import logging
import tempfile
import os
from pathlib import Path

from processors.image_processor import ImageProcessor
from processors.ocr_engine import OCREngineFactory
from processors.omr_logic import OMRProcessor
from processors.validators import FieldValidator, postprocess_text
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SV-20 OCR Mikroservis",
    description="Offline OCR servis za prepoznavanje podataka sa ŠV-20 obrazaca",
    version="1.0.0"
)

class OCRResult(BaseModel):
    field_id: int
    field_name: str
    ocr_value: str
    validated_value: str
    confidence: float
    is_valid: bool
    validation_error: Optional[str] = None

class OCRResponse(BaseModel):
    success: bool
    message: str
    obrazac_id: Optional[str] = None
    total_fields: int
    successful_fields: int
    failed_fields: int
    fields: List[OCRResult]

image_processor = ImageProcessor(target_dpi=Config.TARGET_DPI)
omr_processor = OMRProcessor()
template_path = Config.TEMPLATES_DIR / "sv20_template.json"

def load_template() -> dict:
    """Učitava template za SV-20 obrazac."""
    with open(template_path, 'r', encoding='utf-8') as f:
        return json.load(f)

@app.on_event("startup")
async def startup_event():
    """Inicijalizacija pri pokretanju servera."""
    logger.info("Inicijalizacija OCR Engine-a...")
    OCREngineFactory.get_instance(
        languages=Config.OCR_LANGUAGES,
        model_storage_directory=str(Config.MODELS_DIR),
        use_gpu=Config.USE_GPU,
        gpu_backend=Config.GPU_BACKEND
    )
    logger.info("OCR Engine spreman.")

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "gpu_available": Config.detect_gpu() != "cpu",
        "gpu_backend": Config.detect_gpu()
    }

@app.get("/api/template")
async def get_template():
    """Vraća trenutni template za SV-20 obrazac."""
    try:
        template = load_template()
        return template
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ocr/process", response_model=OCRResponse)
async def process_image(
    file: UploadFile = File(...),
    obrazac_id: Optional[str] = None
):
    """
    Procesira sliku SV-20 obrasca i vraća ekstrahovane podatke.

    - **file**: Slika obrasca (JPG, PNG, PDF)
    - **obrazac_id**: Opcioni ID obrasca za praćenje

    Vraća JSON sa ekstrahovanim poljima, confidence skorovima i validacijom.
    """
    logger.info(f"Primljen zahtev za OCR: {file.filename}")

    allowed_extensions = {'.jpg', '.jpeg', '.png', '.pdf'}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Nepodržan format fajla. Dozvoljeni: {allowed_extensions}"
        )

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        template = load_template()

        img = image_processor.load_image(tmp_path)
        img = image_processor.deskew(img)

        ocr_engine = OCREngineFactory.get_instance()

        results = []
        successful = 0
        failed = 0

        for field in template['fields']:
            try:
                field_result = process_field(img, field, ocr_engine)
                results.append(field_result)

                if field_result.is_valid:
                    successful += 1
                else:
                    failed += 1

            except Exception as e:
                logger.error(f"Greška pri obradi polja {field['name']}: {e}")
                results.append(OCRResult(
                    field_id=field['id'],
                    field_name=field['name'],
                    ocr_value="",
                    validated_value="",
                    confidence=0.0,
                    is_valid=False,
                    validation_error=str(e)
                ))
                failed += 1

        return OCRResponse(
            success=True,
            message="Obrada završena",
            obrazac_id=obrazac_id,
            total_fields=len(template['fields']),
            successful_fields=successful,
            failed_fields=failed,
            fields=results
        )

    except Exception as e:
        logger.error(f"Greška pri OCR obradi: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)

def process_field(img, field_config: dict, ocr_engine) -> OCRResult:
    """
    Procesira pojedinačno polje iz obrasca.
    """
    field_type = field_config.get('type', 'TEXT')

    processed_roi = image_processor.process_field(img, field_config)

    if field_type.startswith('OMR'):
        omr_options = field_config.get('omr_options', [])
        omr_result = omr_processor.detect_marked_option(processed_roi, omr_options)

        ocr_value = omr_result.get('value', '') or ''
        confidence = omr_result.get('confidence', 0.0)

    else:
        if field_type == 'NUMERIC' and field_config.get('validation', {}).get('length') == 13:
            ocr_value, confidence = ocr_engine.recognize_digits_only(processed_roi)
        else:
            ocr_value, confidence = ocr_engine.recognize_single_field(processed_roi)

    postprocess_ops = field_config.get('postprocessing', [])
    if postprocess_ops:
        ocr_value = postprocess_text(ocr_value, postprocess_ops)

    validation_result = FieldValidator.validate_field(ocr_value, field_config)

    return OCRResult(
        field_id=field_config['id'],
        field_name=field_config['name'],
        ocr_value=ocr_value,
        validated_value=validation_result['cleaned_value'],
        confidence=confidence,
        is_valid=validation_result['is_valid'],
        validation_error=validation_result['error']
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=Config.HOST, port=Config.PORT)
```

### 3.8 download_models.py - Offline Priprema

```python
"""
Skripta za preuzimanje svih potrebnih modela unapred.
Pokrenuti jednom pre offline korišćenja.
"""

import os
import sys
from pathlib import Path
import urllib.request
import zipfile
import hashlib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / "models"

EASYOCR_MODELS = {
    "craft_mlt_25k.pth": {
        "url": "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/craft_mlt_25k.zip",
        "size_mb": 90,
        "description": "Text Detection Model"
    },
    "latin_g2.pth": {
        "url": "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/latin_g2.zip",
        "size_mb": 107,
        "description": "Latin Recognition Model"
    },
    "cyrillic_g2.pth": {
        "url": "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/cyrillic_g2.zip",
        "size_mb": 107,
        "description": "Cyrillic Recognition Model"
    }
}

def create_directories():
    """Kreira potrebne direktorijume."""
    dirs = [
        MODELS_DIR,
        MODELS_DIR / "sr_cyrl",
        MODELS_DIR / "sr_latn",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Kreiran direktorijum: {d}")

def download_file(url: str, dest_path: Path) -> bool:
    """Preuzima fajl sa progress indikatorom."""
    try:
        logger.info(f"Preuzimanje: {url}")

        def progress_hook(count, block_size, total_size):
            percent = min(100, int(count * block_size * 100 / total_size))
            sys.stdout.write(f"\r  Progress: {percent}%")
            sys.stdout.flush()

        urllib.request.urlretrieve(url, dest_path, progress_hook)
        print()
        logger.info(f"Sačuvano u: {dest_path}")
        return True

    except Exception as e:
        logger.error(f"Greška pri preuzimanju: {e}")
        return False

def extract_zip(zip_path: Path, dest_dir: Path):
    """Ekstraktuje ZIP arhivu."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
        logger.info(f"Ekstraktovano u: {dest_dir}")
        os.unlink(zip_path)
    except Exception as e:
        logger.error(f"Greška pri ekstraktovanju: {e}")

def download_easyocr_models():
    """Preuzima EasyOCR modele."""
    logger.info("=" * 50)
    logger.info("Preuzimanje EasyOCR modela...")
    logger.info("=" * 50)

    for model_name, model_info in EASYOCR_MODELS.items():
        model_path = MODELS_DIR / model_name

        if model_path.exists():
            logger.info(f"Model već postoji: {model_name}")
            continue

        logger.info(f"\nModel: {model_info['description']}")
        logger.info(f"Veličina: ~{model_info['size_mb']} MB")

        zip_path = MODELS_DIR / f"{model_name}.zip"

        if download_file(model_info['url'], zip_path):
            extract_zip(zip_path, MODELS_DIR)

def verify_installation():
    """Verifikuje da su svi modeli preuzeti."""
    logger.info("\n" + "=" * 50)
    logger.info("Verifikacija instalacije...")
    logger.info("=" * 50)

    all_ok = True
    for model_name in EASYOCR_MODELS.keys():
        model_path = MODELS_DIR / model_name
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            logger.info(f"✓ {model_name} ({size_mb:.1f} MB)")
        else:
            logger.error(f"✗ {model_name} - NEDOSTAJE!")
            all_ok = False

    return all_ok

def test_ocr():
    """Testira OCR sa demo slikom."""
    logger.info("\n" + "=" * 50)
    logger.info("Testiranje OCR Engine-a...")
    logger.info("=" * 50)

    try:
        import easyocr

        reader = easyocr.Reader(
            ['sr_cyrl', 'sr_latn'],
            model_storage_directory=str(MODELS_DIR),
            download_enabled=False,
            gpu=False
        )

        logger.info("✓ OCR Engine uspešno inicijalizovan!")
        logger.info("  Jezici: sr_cyrl, sr_latn")
        logger.info("  Model direktorijum: " + str(MODELS_DIR))

        return True

    except Exception as e:
        logger.error(f"✗ Greška pri testiranju: {e}")
        return False

def main():
    print("""
    ╔══════════════════════════════════════════════════╗
    ║     SV-20 OCR - Offline Model Downloader         ║
    ╠══════════════════════════════════════════════════╣
    ║  Ovaj skript preuzima sve potrebne modele za     ║
    ║  offline rad OCR servisa.                        ║
    ╚══════════════════════════════════════════════════╝
    """)

    create_directories()
    download_easyocr_models()

    if verify_installation():
        if test_ocr():
            print("""
    ╔══════════════════════════════════════════════════╗
    ║           INSTALACIJA USPEŠNA!                   ║
    ╠══════════════════════════════════════════════════╣
    ║  Svi modeli su preuzeti i verifikovani.         ║
    ║  Servis je spreman za offline rad.              ║
    ║                                                  ║
    ║  Pokrenite servis sa: python main.py            ║
    ╚══════════════════════════════════════════════════╝
            """)
        else:
            print("\n⚠ Instalacija završena, ali test nije prošao.")
    else:
        print("\n✗ Neki modeli nedostaju. Pokrenite skriptu ponovo.")

if __name__ == "__main__":
    main()
```

### 3.9 main.py - Entry Point

```python
"""
SV-20 OCR Mikroservis - Entry Point
"""

import uvicorn
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from api.server import app

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOGS_DIR / 'ocr_service.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def check_prerequisites():
    """Proverava da li su svi preduslovi ispunjeni."""

    if not Config.MODELS_DIR.exists():
        logger.error("Models direktorijum ne postoji!")
        logger.error("Pokrenite: python download_models.py")
        return False

    template_path = Config.TEMPLATES_DIR / "sv20_template.json"
    if not template_path.exists():
        logger.error(f"Template fajl ne postoji: {template_path}")
        return False

    return True

def main():
    print("""
    ╔══════════════════════════════════════════════════╗
    ║        SV-20 OCR Mikroservis v1.0                ║
    ╠══════════════════════════════════════════════════╣
    ║  Offline OCR za prepoznavanje podataka           ║
    ║  sa ŠV-20 obrazaca                               ║
    ╚══════════════════════════════════════════════════╝
    """)

    if not check_prerequisites():
        sys.exit(1)

    gpu_backend = Config.detect_gpu()
    logger.info(f"Detektovan backend: {gpu_backend}")

    logger.info(f"Pokrećem server na {Config.HOST}:{Config.PORT}")

    uvicorn.run(
        app,
        host=Config.HOST,
        port=Config.PORT,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main()
```

### 3.10 requirements.txt

```
# Core
fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6

# OCR
easyocr==1.7.1
onnxruntime-directml==1.16.3  # AMD GPU support

# Image Processing
opencv-python==4.8.1.78
numpy==1.24.4
Pillow==10.1.0

# PDF Support
pdf2image==1.16.3
PyMuPDF==1.23.7

# Validation
pydantic==2.5.2

# Utilities
python-dotenv==1.0.0
```

---

## 4. Java Integracija

### 4.1 OCRService.java - Klijentska Strana

```java
package komunikacija;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.UUID;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

public class OCRService {

    private static final String OCR_SERVICE_URL = "http://127.0.0.1:9001";
    private static final int TIMEOUT_SECONDS = 60;

    private final HttpClient httpClient;
    private final Gson gson;

    public OCRService() {
        this.httpClient = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(TIMEOUT_SECONDS))
            .build();
        this.gson = new Gson();
    }

    public boolean isServiceAvailable() {
        try {
            HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(OCR_SERVICE_URL + "/api/health"))
                .GET()
                .build();

            HttpResponse<String> response = httpClient.send(request,
                HttpResponse.BodyHandlers.ofString());

            return response.statusCode() == 200;
        } catch (Exception e) {
            return false;
        }
    }

    public OCRResult processImage(String imagePath, int obrazacId) throws Exception {
        File imageFile = new File(imagePath);
        if (!imageFile.exists()) {
            throw new IOException("Fajl ne postoji: " + imagePath);
        }

        String boundary = UUID.randomUUID().toString();

        byte[] fileBytes = Files.readAllBytes(Path.of(imagePath));
        String fileName = imageFile.getName();

        String bodyStart = "--" + boundary + "\r\n" +
            "Content-Disposition: form-data; name=\"file\"; filename=\"" + fileName + "\"\r\n" +
            "Content-Type: application/octet-stream\r\n\r\n";

        String bodyEnd = "\r\n--" + boundary + "\r\n" +
            "Content-Disposition: form-data; name=\"obrazac_id\"\r\n\r\n" +
            obrazacId + "\r\n--" + boundary + "--\r\n";

        byte[] body = concatenateArrays(
            bodyStart.getBytes(),
            fileBytes,
            bodyEnd.getBytes()
        );

        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(OCR_SERVICE_URL + "/api/ocr/process"))
            .header("Content-Type", "multipart/form-data; boundary=" + boundary)
            .POST(HttpRequest.BodyPublishers.ofByteArray(body))
            .timeout(Duration.ofSeconds(TIMEOUT_SECONDS))
            .build();

        HttpResponse<String> response = httpClient.send(request,
            HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() != 200) {
            throw new Exception("OCR greška: " + response.body());
        }

        return parseResponse(response.body());
    }

    private OCRResult parseResponse(String json) {
        JsonObject root = gson.fromJson(json, JsonObject.class);

        OCRResult result = new OCRResult();
        result.setSuccess(root.get("success").getAsBoolean());
        result.setMessage(root.get("message").getAsString());
        result.setTotalFields(root.get("total_fields").getAsInt());
        result.setSuccessfulFields(root.get("successful_fields").getAsInt());
        result.setFailedFields(root.get("failed_fields").getAsInt());

        JsonArray fields = root.getAsJsonArray("fields");
        for (int i = 0; i < fields.size(); i++) {
            JsonObject field = fields.get(i).getAsJsonObject();

            OCRFieldResult fieldResult = new OCRFieldResult();
            fieldResult.setFieldId(field.get("field_id").getAsInt());
            fieldResult.setFieldName(field.get("field_name").getAsString());
            fieldResult.setOcrValue(field.get("ocr_value").getAsString());
            fieldResult.setValidatedValue(field.get("validated_value").getAsString());
            fieldResult.setConfidence(field.get("confidence").getAsDouble());
            fieldResult.setValid(field.get("is_valid").getAsBoolean());

            if (field.has("validation_error") && !field.get("validation_error").isJsonNull()) {
                fieldResult.setValidationError(field.get("validation_error").getAsString());
            }

            result.addField(fieldResult);
        }

        return result;
    }

    private byte[] concatenateArrays(byte[]... arrays) {
        int totalLength = 0;
        for (byte[] arr : arrays) {
            totalLength += arr.length;
        }

        byte[] result = new byte[totalLength];
        int offset = 0;
        for (byte[] arr : arrays) {
            System.arraycopy(arr, 0, result, offset, arr.length);
            offset += arr.length;
        }

        return result;
    }
}
```

---

## 5. Deployment Checklist

### Pre Odbrane Seminarskog:

```
□ 1. Pokrenuti download_models.py na računaru za odbranu
□ 2. Verifikovati da su svi modeli preuzeti (~300 MB)
□ 3. Testirati sa sample SV-20 obrascem
□ 4. Proveriti da nema network poziva (wireshark/tcpdump)
□ 5. Dokumentovati GPU detekciju (AMD DirectML)
□ 6. Pripremiti fallback na CPU ako GPU ne radi
□ 7. Testirati integraciju sa Java klijentom
□ 8. Pripremiti 2-3 test obrasca različitog kvaliteta
```

### Pokretanje Sistema:

```bash
# Terminal 1 - Python OCR Servis
cd sv20-ocr-service
python main.py

# Terminal 2 - Java Server
cd Prosoft/SERVER
java -jar SERVER.jar

# Terminal 3 - Java Klijent
cd Prosoft/KLIJENT
java -jar KLIJENT.jar
```

---

## 6. Dijagram Toka Obrade

```
┌─────────────────┐
│   Slika SV-20   │
│   (.jpg/.png)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Deskew        │  ← Ispravljanje nagiba
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Za svako polje: │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐
│ TEXT  │ │  OMR  │
└───┬───┘ └───┬───┘
    │         │
    ▼         ▼
┌───────┐ ┌───────┐
│Denoise│ │Pixel  │
│ROI    │ │Density│
│OCR    │ │Check  │
└───┬───┘ └───┬───┘
    │         │
    └────┬────┘
         │
         ▼
┌─────────────────┐
│   Validacija    │
│ (JMBG, Datum...)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   JSON Response │
│   → Java Klijent│
└─────────────────┘
```

---

## 7. Očekivani Rezultati

### Primer JSON Response-a:

```json
{
  "success": true,
  "message": "Obrada završena",
  "obrazac_id": "SV20-2025-001",
  "total_fields": 12,
  "successful_fields": 11,
  "failed_fields": 1,
  "fields": [
    {
      "field_id": 1,
      "field_name": "ime_studenta",
      "ocr_value": "Вукашин",
      "validated_value": "Вукашин",
      "confidence": 0.94,
      "is_valid": true,
      "validation_error": null
    },
    {
      "field_id": 3,
      "field_name": "jmbg",
      "ocr_value": "0101998710029",
      "validated_value": "0101998710029",
      "confidence": 0.89,
      "is_valid": true,
      "validation_error": null
    },
    {
      "field_id": 5,
      "field_name": "pol",
      "ocr_value": "M",
      "validated_value": "M",
      "confidence": 0.92,
      "is_valid": true,
      "validation_error": null
    }
  ]
}
```

Naslov: Implementacija Offline Python OCR Mikroservisa za ŠV-20 (AMD GPU/DirectML) sa Dinamičkim Mapiranjem Polja

Zadatak: Napravi kompletan plan i implementaciju Python OCR mikroservisa koji će Java klijent pozivati lokalno. Fokus je na ekstremnoj preciznosti pogađanja koordinata polja na skeniranim ŠV-20 obrascima koristeći napredne OpenCV tehnike.

1. Strategija Preciznog Pozicioniranja (Pogađanje koordinata):

Perspective Transform: Implementiraj modul koji detektuje ivice papira na slici i vrši "ispravljanje" (warp perspective) dokumenta na standardizovanu A4 proporciju. Ovo osigurava da su polja uvek na predvidljivim mestima bez obzira na ugao skeniranja.

Anchor-Based Alignment: Umesto fiksnih piksela, koristi Template Matching za detekciju fiksnih elemenata (sidara) kao što je logotip "РЗС" i naslov "Образац ШВ-20". Sve koordinate polja u sv20_template.json treba da budu relativne u odnosu na ova sidra.

Dinamički OMR: Za zaokružene odgovore (Pol, Status, Finansiranje), implementiraj Hough Circle Transform za detekciju rukom nacrtanih krugova oko ponuđenih brojeva/opcija.

2. Tehnički Stack i Hardver:

AMD Radeon Optimizacija: Koristiti onnxruntime-directml za GPU ubrzanje.

Jezici: EasyOCR konfigurisati za istovremeno čitanje sr_cyrl i sr_latn.

Backend: FastAPI na portu 9001.

3. Moduli za Implementaciju:

image_processor.py: Sadrži deskew, perspective_transform, line_removal (za brisanje linija kućica JMBG-a) i anchor_detection.

ocr_engine.py: EasyOCR wrapper sa ONNX podrškom i optimizacijom za AMD.

omr_logic.py: Detekcija zaokruženih polja putem analize gustine piksela i kontura.

validators.py: Logika za proveru JMBG-a (13 cifara i kontrolni broj) i formata indeksa (GGGG/BBBB).

4. Specifični zahtevi za ŠV-20:

JMBG: Poseban tretman sečenja svake kućice (13 segmenata) radi maksimalne preciznosti cifara.

Offline Setup: Napravi download_models.py skriptu koja preuzima sve modele unapred kako bi sistem radio bez interneta na odbrani seminarskog.

JSON Response: Vratiti detaljan JSON sa tekstom, očišćenom vrednošću, confidence skorom i statusom validacije za sva@ko polje.