"""
SV-20 OCR Mikroservis - Handwriting OCR (TrOCR, eksperimentalno, WS6)

Opcioni "drugo mišljenje" OCR engine za rukom pisana polja, iza feature-flaga
(Config.ENABLE_HANDWRITING_TROCR, default False). EasyOCR (processors/ocr_engine.py)
ostaje primarni engine za SVA polja - ovaj modul se poziva SAMO kad je flag
uključen I (polje je eksplicitno označeno kao "handwriting": true u
template-u ILI je primarni EasyOCR confidence ispod
Config.HANDWRITING_FALLBACK_CONFIDENCE).

VAŽNO - stvarna tačnost na srpskom rukopisu NIJE potvrđena unapred: ne
postoji gotov TrOCR model treniran specifično za srpski rukopis. Koristi se
najbliži dostupan - "cyrillic-trocr/trocr-handwritten-cyrillic" (HuggingFace),
treniran na ruskom/ukrajinskom/crkvenoslovenskom rukopisu. Srpska ćirilica
deli većinu grafema sa ruskom, ali NEMA u tom skupu slova Ђ/Ћ/Џ/Љ/Њ - realan
rizik za greške baš na njima. Za latinicu ne postoji dobar gotov model za
srpske dijakritike (š/č/ć/ž/đ). Testiraj na svojim skenovima (upoređujući
ocr_value vs alt_value u JSON odgovoru) pre nego što veruješ rezultatima -
vidi ANALIZA-OCR-SERVISA.md, radni paket WS6.
"""

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "cyrillic-trocr/trocr-handwritten-cyrillic"


class HandwritingOCREngine:
    """
    Lenjo (lazy) učitavanje TrOCR modela - `transformers`/`torch` model se
    učitava tek pri PRVOM stvarnom pozivu recognize(), ne pri importu modula
    niti pri startu servisa, tako da servis normalno radi i bez ove (opcione,
    relativno teške ~500MB) zavisnosti kad je ENABLE_HANDWRITING_TROCR
    isključen (default stanje).
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self._processor = None
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return

        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError as e:
            raise ImportError(
                "TrOCR fallback (Config.ENABLE_HANDWRITING_TROCR) zahteva 'transformers' "
                "paket koji nije instaliran. Instaliraj sa: pip install transformers"
            ) from e

        logger.info(f"[TrOCR] Učitavam model '{self.model_name}' (prvi poziv - može potrajati "
                   f"i preuzeti ~500MB pri prvom pokretanju)...")
        self._processor = TrOCRProcessor.from_pretrained(self.model_name)
        self._model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
        self._model.eval()
        logger.info("[TrOCR] Model spreman.")

    def recognize(self, image: np.ndarray) -> Tuple[str, float]:
        """
        OCR jednog ROI-ja preko TrOCR-a.

        Args:
            image: ROI slike polja (OpenCV BGR ili grayscale numpy array)

        Returns:
            (tekst, confidence) - confidence je geometrijska sredina
            verovatnoća pojedinačno generisanih tokena (standardna mera
            pouzdanosti za seq2seq generisanje - ekvivalentno inverznoj
            perplexity-ju modela na tu sekvencu), ne izmišljena konstanta.
        """
        self._ensure_loaded()

        try:
            import torch
            import cv2
            from PIL import Image

            if len(image.shape) == 2:
                rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 3:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb = image

            pil_image = Image.fromarray(rgb)
            pixel_values = self._processor(images=pil_image, return_tensors="pt").pixel_values

            with torch.no_grad():
                outputs = self._model.generate(
                    pixel_values,
                    max_length=64,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            text = self._processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
            confidence = self._sequence_confidence(outputs)

            return text.strip(), confidence

        except Exception as e:
            logger.error(f"[TrOCR] Greška pri prepoznavanju: {e}")
            return "", 0.0

    @staticmethod
    def _sequence_confidence(outputs) -> float:
        """
        Geometrijska sredina verovatnoća generisanih tokena (standardna mera
        pouzdanosti za seq2seq generisanje, koristi se u literaturi kao
        aproksimacija verovatnoće cele sekvence bez favorizovanja kraćih
        odgovora - obična suma log-verovatnoća bi to radila).
        """
        import torch

        scores = getattr(outputs, "scores", None)
        sequences = getattr(outputs, "sequences", None)
        if not scores or sequences is None:
            return 0.0

        log_probs = []
        for step, step_logits in enumerate(scores):
            step_log_probs = torch.log_softmax(step_logits[0], dim=-1)
            token_id = sequences[0, step + 1]
            log_probs.append(step_log_probs[token_id].item())

        if not log_probs:
            return 0.0

        mean_log_prob = sum(log_probs) / len(log_probs)
        return float(min(max(np.exp(mean_log_prob), 0.0), 1.0))


class HandwritingOCRFactory:
    """Singleton instanca, isti obrazac kao OCREngineFactory (ocr_engine.py)."""

    _instance: Optional[HandwritingOCREngine] = None

    @classmethod
    def get_instance(cls) -> HandwritingOCREngine:
        if cls._instance is None:
            cls._instance = HandwritingOCREngine()
        return cls._instance

    @classmethod
    def reset(cls):
        cls._instance = None
