"""
SV-20 OCR Mikroservis - OMR Logic
Optical Mark Recognition - detekcija ZAOKRUŽENIH BROJEVA.

SV-20 obrazac koristi zaokruživanje brojeva (circled numbers), a ne
popunjavanje kružića (bubbles). Detekcija se oslanja na PRECIZNE koordinate
opcija sačuvane preko Template Editora (static/index.html) - fill_ratio i
edge_score po opciji, sa statistički izvedenom (ne hardkodovanom) confidence
merom (vidi _relative_margin_confidence). Za istoriju starijih, uklonjenih
heuristika (contour/edge/zone density bez preciznih koordinata) vidi
ANALIZA-OCR-SERVISA.md, stavka 4.6.
"""

import cv2
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class OMRProcessor:
    """
    Optical Mark Recognition - detekcija zaokruženih brojeva na osnovu
    preciznih koordinata opcija (Template Editor).
    """

    def detect_marked_option(self,
                             image: np.ndarray,
                             options: List[Dict],
                             method: str = "auto") -> Dict:
        """
        Glavna metoda za detekciju označene opcije.

        Ako opcije imaju precizne x,y koordinate (iz Template Editora), koristi
        statističku analizu ZASNOVANU ISKLJUČIVO NA OPCIJAMA OVOG POLJA -
        razmerni (ratio) margin između najjače i druge najjače opcije (videti
        `_relative_margin_confidence`). Namerno se NE upoređuje sa drugim
        poljima na strani: opcije različitih polja na ŠV-20 obrascu imaju
        veoma različitu "prirodnu" količinu odštampanog mastila (npr. polje
        sa dugim tekstualnim opcijama nasuprot polju sa dva mala kružića), pa
        bi mešanje tih skala u jednu statistiku davalo pogrešne rezultate -
        ovo je potvrđeno na stvarnim skenovima, ne pretpostavka.

        Bez preciznih koordinata (redak slučaj - polje ručno dodato u JSON
        mimo Editora) vraća value=None sa jasnim upozorenjem, vidi ispod.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape[:2]

        if not options:
            return {'value': None, 'confidence': 0.0, 'warning': 'Nema definisanih opcija'}

        # PROVERA: Da li opcije imaju precizne koordinate (x, y)?
        has_precise_coords = all('x' in opt and 'y' in opt for opt in options)

        if has_precise_coords:
            # NE pada dalje na legacy contour/edge/zone heuristike kad je
            # confidence niska - to bi samo zamenilo dobro kalibrisan "nema
            # oznake" odgovor lošije kalibrisanim nagađanjem starijih metoda.
            # Legacy lanac ostaje samo za slučaj kad uopšte NEMA preciznih
            # koordinata.
            logger.info("[OMR] Koristim precizne koordinate krugova za statističku analizu")
            option_scores = self.extract_option_scores(gray, options)
            result = self._decide_single(option_scores)
            logger.info(f"[OMR] Precise coords: {result['value']} (conf={result['confidence']:.2f})")
            return result

        # Bez preciznih (x,y) koordinata opcija OMR detekcija nije podržana.
        # Do ovde su ranije postojale 3 fallback heuristike (contour-krug,
        # edge-density, zone-density) sa desetinama nezavisno "nameštenih"
        # pragova bez ijednog regresionog testa - uklonjene su (ANALIZA-OCR-
        # SERVISA.md 4.6) jer su u praksi bile mrtav kod: Template Editor
        # UVEK čuva precizne x,y koordinate za svaku opciju (proverено na
        # svih 12 OMR polja u templates/sv20_template.json), pa se ova grana
        # realno nikad nije izvršavala. Ako se ovo javi, znači da je neko
        # polje ručno dodato u JSON bez korišćenja Editora.
        logger.warning(f"[OMR] Polje nema precizne (x,y) koordinate opcija - definiši ih u "
                      f"Template Editoru (Uredi krugove). OMR detekcija bez toga nije podržana.")
        return {
            'value': None,
            'confidence': 0.0,
            'warning': 'Opcije nemaju precizne koordinate (x,y) - definiši ih u Template Editoru',
            'method': 'no_precise_coords'
        }

    def extract_option_scores(self, gray: np.ndarray, options: List[Dict]) -> List[Dict]:
        """
        Čista ekstrakcija signala po opciji (fill_ratio, edge_score,
        combined_score) - BEZ ikakve odluke o tome šta je "označeno". Tu
        odluku prave `_decide_single`/`_decide_multi` na osnovu statistike
        (medijana/MAD) cele strane, ne ova funkcija - videti
        `compute_global_stats`.

        Algoritam po opciji:
        1. Izvuci ROI (x, y, w, h) - CELE dimenzije koje je korisnik namestio
           u Template Editoru, bez smanjivanja.
        2. Binarizuj (adaptivni threshold + blur da spoji linije zaokruživanja).
        3. combined_score = fill_ratio * 0.6 + edge_score * 0.4
           (fill_ratio = % "mastila" u ROI-ju, edge_score = da li ivice ROI-ja
           imaju kružnu strukturu - vidi _analyze_circle_edges)
        """
        logger.info(f"[OMR SCORES] Analiziram {len(options)} opcija sa preciznim koordinatama")

        # Binarizacija - crna pozadina, belo "mastilo"
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Malo blur-a da spojimo linije zaokruživanja
        binary = cv2.GaussianBlur(binary, (3, 3), 0)
        _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)

        h, w = gray.shape[:2]
        results = []

        for i, option in enumerate(options):
            x = int(option.get('x', 0))
            y = int(option.get('y', 0))
            opt_w = int(option.get('width', 40))
            opt_h = int(option.get('height', 40))

            # FULL RECTANGLE - koristi CELE dimenzije koje je korisnik namestio u editoru
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w, x + opt_w)
            y2 = min(h, y + opt_h)

            if x2 <= x1 or y2 <= y1:
                logger.warning(f"[OMR SCORES] Opcija {option.get('label')} ima nevalidne koordinate: ({x1},{y1})-({x2},{y2})")
                continue

            roi = binary[y1:y2, x1:x2]

            white_pixels = cv2.countNonZero(roi)
            total_pixels = roi.shape[0] * roi.shape[1]
            fill_ratio = white_pixels / total_pixels if total_pixels > 0 else 0

            edge_score = self._analyze_circle_edges(roi)
            combined_score = fill_ratio * 0.6 + edge_score * 0.4

            results.append({
                'label': option.get('label', str(i + 1)),
                'value': option.get('value', option.get('label', str(i + 1))),
                'fill_ratio': fill_ratio,
                'edge_score': edge_score,
                'combined_score': combined_score,
                'roi_size': (opt_w, opt_h)
            })

            logger.debug(f"[OMR SCORES] {option.get('label')}: fill={fill_ratio:.3f}, edge={edge_score:.3f}, combined={combined_score:.3f}")

        return results

    @staticmethod
    def _relative_margin_confidence(score: float, reference: float) -> float:
        """
        confidence = koliki deo SOPSTVENOG score-a opcija ima "viška" u
        odnosu na referentnu (drugu najjaču, ili najslabiju - zavisno od
        poziva) vrednost:

            confidence = (score - reference) / score      za score > 0
            confidence = 0                                 za score <= 0

        Ovo je namerno RAZMERNA (ratio), ne ADITIVNA mera, i namerno je
        potpuno LOKALNA za samo jedno polje (bez poređenja preko više polja
        na strani). Razlog: na ŠV-20 obrascu opcije različitih polja imaju
        drastično različitu "prirodnu" količinu odštampanog mastila (npr.
        polje sa dugačkim tekstualnim opcijama nasuprot polju sa samo dva
        malа kružića) - aditivna statistika (npr. medijana/MAD) izračunata
        preko VIŠE polja meša te nesamerljive skale i daje pogrešne rezultate
        (ovo je izmereno i potvrđeno na stvarnom skenu, ne pretpostavka).
        Razmerna mera unutar JEDNOG polja je skalо-nezavisna: ista formula
        daje ispravan rezultat i za polje čije su vrednosti ~0.01 i za polje
        čije su vrednosti ~0.30, bez ijedne konstante koja bi morala da se
        podešava po tipu polja.

        confidence = 0.5 prirodno znači "pobednik ima tačno duplo više
        signala od reference" (score = 2 * reference) - simetrična, lako
        objašnjiva tačka, ne broj biran "na oko".
        """
        if score <= 0:
            return 0.0
        return max(0.0, min((score - reference) / score, 1.0))

    def _decide_single(self, option_scores: List[Dict]) -> Dict:
        """
        Bira najbolju opciju (single-select) unutar JEDNOG polja, poredeći je
        sa drugom najjačom opcijom istog polja (vidi _relative_margin_confidence).
        """
        if not option_scores:
            return {
                'value': None,
                'confidence': 0.0,
                'warning': 'Nema validnih opcija',
                'method': 'precise_coords'
            }

        ranked = sorted(option_scores, key=lambda x: x['combined_score'], reverse=True)
        best = ranked[0]
        second_best_score = ranked[1]['combined_score'] if len(ranked) > 1 else 0.0

        confidence = self._relative_margin_confidence(best['combined_score'], second_best_score)

        for r in option_scores:
            other_max = max(
                [o['combined_score'] for o in option_scores if o is not r],
                default=0.0
            )
            r['confidence'] = round(self._relative_margin_confidence(r['combined_score'], other_max), 3)

        if confidence <= 0.5:
            logger.info(f"[OMR PRECISE] Nema ubedljivo označene opcije (najbolja '{best['label']}', conf={confidence:.2f})")
            return {
                'value': None,
                'confidence': round(confidence, 3),
                'warning': 'Nijedna opcija nije dovoljno pouzdano označena',
                'method': 'precise_coords',
                'all_options': option_scores
            }

        logger.info(f"[OMR PRECISE] ✓ Detekcija: {best['label']} = {best['value']} (conf={confidence:.2f})")

        return {
            'value': best['value'],
            'label': best['label'],
            'confidence': round(confidence, 3),
            'fill_ratio': best['fill_ratio'],
            'edge_score': best['edge_score'],
            'combined_score': best['combined_score'],
            'method': 'precise_coords',
            'all_options': option_scores
        }

    def _decide_multi(self, option_scores: List[Dict]) -> Dict:
        """
        Nezavisna odluka po opciji (multi-select) unutar JEDNOG polja - svaka
        opcija se poredi sa najslabijom ("najpraznijom") opcijom istog polja
        kao lokalnom referencom za "prazno" (vidi _relative_margin_confidence).
        """
        if not option_scores:
            return {'values': [], 'count': 0, 'confidence': 0.0, 'method': 'multi_select', 'all_options': []}

        floor = min(o['combined_score'] for o in option_scores)

        detected_values = []
        results = []

        for opt in option_scores:
            confidence = self._relative_margin_confidence(opt['combined_score'], floor)
            is_detected = confidence > 0.5
            enriched = {**opt, 'confidence': round(confidence, 3), 'detected': is_detected}
            results.append(enriched)

            logger.debug(f"[OMR MULTI PRECISE] {opt.get('label')}: combined={opt['combined_score']:.3f} conf={confidence:.2f} -> {'✓' if is_detected else '✗'}")

            if is_detected:
                detected_values.append(opt['value'])

        overall_confidence = max([r['confidence'] for r in results if r['detected']], default=0.0)

        logger.info(f"[OMR MULTI PRECISE] ✓ Detektovano {len(detected_values)} opcija: {detected_values}")

        return {
            'values': detected_values,
            'count': len(detected_values),
            'confidence': round(overall_confidence, 3),
            'method': 'multi_select',
            'all_options': results
        }

    def _analyze_circle_edges(self, roi: np.ndarray) -> float:
        """
        Analizira da li ROI sadrži kružnu strukturu (zaokruživanje).

        Vraća score 0-1 baziran na tome koliko ivica postoji na periferiji ROI-a.
        Zaokružen broj ima linije oko sebe, nezaokružen nema.
        """
        h, w = roi.shape[:2]

        if h < 10 or w < 10:
            return 0.0

        # Edge detection
        edges = cv2.Canny(roi, 30, 100)

        # Podeli ROI na centar i periferiju
        margin_h = max(1, h // 5)
        margin_w = max(1, w // 5)

        # Periferija = gornja/donja/leva/desna ivica
        top = edges[0:margin_h, :]
        bottom = edges[h-margin_h:h, :]
        left = edges[:, 0:margin_w]
        right = edges[:, w-margin_w:w]

        periphery_pixels = (
            cv2.countNonZero(top) +
            cv2.countNonZero(bottom) +
            cv2.countNonZero(left) +
            cv2.countNonZero(right)
        )

        periphery_size = top.size + bottom.size + left.size + right.size

        if periphery_size == 0:
            return 0.0

        periphery_density = periphery_pixels / periphery_size

        # Normalizuj na 0-1
        # Zaokruženje obično ima density 0.05-0.3
        score = min(periphery_density / 0.2, 1.0)

        return score

    def detect_multi_select(self,
                            image: np.ndarray,
                            options: List[Dict]) -> Dict:
        """Detektuje više označenih opcija. Vidi detect_marked_option za objašnjenje pristupa."""
        gray = self._to_grayscale(image)
        h, w = gray.shape[:2]

        # NOVA LOGIKA: Ako opcije imaju precizne koordinate, koristi statističku analizu
        has_precise_coords = all('x' in opt and 'y' in opt for opt in options)

        if has_precise_coords:
            logger.info("[OMR MULTI] Koristim precizne koordinate za multi-select detekciju")
            option_scores = self.extract_option_scores(gray, options)
            return self._decide_multi(option_scores)

        # Bez preciznih koordinata multi-select detekcija nije podržana - vidi
        # objašnjenje u detect_marked_option (ANALIZA-OCR-SERVISA.md 4.6).
        logger.warning("[OMR MULTI] Polje nema precizne (x,y) koordinate opcija - definiši ih u Template Editoru")
        return {
            'values': [],
            'count': 0,
            'confidence': 0.0,
            'method': 'no_precise_coords',
            'warning': 'Opcije nemaju precizne koordinate (x,y) - definiši ih u Template Editoru'
        }

    def _to_grayscale(self, img: np.ndarray) -> np.ndarray:
        """Konvertuje u grayscale."""
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _crop_roi(self, image: np.ndarray, coords: Dict) -> np.ndarray:
        """Iseca ROI."""
        x = coords.get('x', 0)
        y = coords.get('y', 0)
        w = coords.get('width', image.shape[1] if len(image.shape) >= 2 else 100)
        h = coords.get('height', image.shape[0] if len(image.shape) >= 1 else 100)

        h_img, w_img = image.shape[:2]
        x = max(0, min(x, w_img))
        y = max(0, min(y, h_img))
        x2 = max(0, min(x + w, w_img))
        y2 = max(0, min(y + h, h_img))

        return image[y:y2, x:x2]


# Debug funkcija za snimanje ROI-a
def save_debug_roi(image: np.ndarray, field_name: str, output_dir: str = "debug_rois"):
    """Sačuvaj ROI za debugging."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{output_dir}/{field_name}.png"
    cv2.imwrite(filename, image)
    logger.info(f"Debug ROI saved: {filename}")


# Testovi: sv20-ocr-service/tests/test_omr_logic.py (pytest) - koriste
# precizne koordinate opcija, kao i stvarni pipeline preko Template Editora.
