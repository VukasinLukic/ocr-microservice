"""
SV-20 OCR Mikroservis - OMR Logic (Contour-Based Circle Detection)
Optical Mark Recognition - detekcija ZAOKRUŽENIH BROJEVA.

SV-20 obrazac koristi zaokruživanje brojeva (circled numbers),
a ne popunjavanje kružića (bubbles). Ova verzija koristi detekciju
kontura za pronalaženje ručno nacrtanih krugova.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class OMRProcessor:
    """
    Optical Mark Recognition - detekcija zaokruženih brojeva.

    NOVI PRISTUP: Contour-Based Circle Detection
    1. Pronađi sve konture u ROI-u
    2. Filtriraj konture koje izgledaju kao ručno nacrtani krugovi
    3. Za svaki detektovani krug, odredi u kojoj je zoni (koja opcija)
    4. Vrati opciju koja ima krug oko sebe
    """

    def __init__(self,
                 fill_threshold: float = 0.15,
                 min_contour_area: int = 200,
                 max_contour_area: int = 50000,
                 circularity_threshold: float = 0.3):
        """
        Args:
            fill_threshold: Procenat popunjenosti za fallback detekciju
            min_contour_area: Minimalna površina konture za krug
            max_contour_area: Maksimalna površina konture
            circularity_threshold: Minimalna "kružnost" konture (0-1)
        """
        self.fill_threshold = fill_threshold
        self.min_contour_area = min_contour_area
        self.max_contour_area = max_contour_area
        self.circularity_threshold = circularity_threshold

    def detect_marked_option(self,
                             image: np.ndarray,
                             options: List[Dict],
                             method: str = "auto") -> Dict:
        """
        Glavna metoda za detekciju označene opcije.

        NOVI PRISTUP (v3): Ako opcije imaju precizne x,y koordinate,
        koristi jednostavnu fill_ratio analizu - "ko ima najviše mastila pobjeđuje"

        Inače, fallback na stare metode (contour, edge, zone).
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
            # NOVA METODA: Koristi precizne koordinate za fill_ratio analizu
            logger.info("[OMR] Koristim precizne koordinate krugova za fill_ratio analizu")
            result = self._detect_by_precise_coords(gray, options)

            if result['confidence'] > 0.3:
                logger.info(f"[OMR] Precise coords: {result['value']} (conf={result['confidence']:.2f}, fill={result.get('fill_ratio', 0):.3f})")
                return result
            else:
                logger.warning(f"[OMR] Precise coords low confidence ({result['confidence']:.2f}), fallback...")

        # FALLBACK: Pripremi opcije za stare metode
        options = self._prepare_options(options, h, w)

        # METODA 1: Contour-based detekcija krugova
        circle_result = self._detect_circles_by_contour(gray, options)

        if circle_result['confidence'] > 0.6:
            logger.info(f"Contour circle detection: {circle_result['value']} (conf={circle_result['confidence']:.2f})")
            return circle_result

        # METODA 2: Edge-based analiza (traži "deblje" linije od zaokruživanja)
        edge_result = self._detect_by_edge_density(gray, options)

        if edge_result['confidence'] > 0.5:
            logger.info(f"Edge density detection: {edge_result['value']} (conf={edge_result['confidence']:.2f})")
            return edge_result

        # METODA 3: Zone-based ink density (fallback)
        density_result = self._detect_by_zone_density(gray, options)

        logger.info(f"Zone density fallback: {density_result['value']} (conf={density_result['confidence']:.2f})")
        return density_result

    def _detect_by_precise_coords(self, gray: np.ndarray, options: List[Dict]) -> Dict:
        """
        NOVA METODA v3: Koristi precizne x, y, width, height koordinate krugova.

        Algoritam:
        1. Za svaki krug izvuci ROI (x, y, w, h)
        2. Binarizuj
        3. Izračunaj fill_ratio (procenat crnih piksela)
        4. Onaj sa NAJVEĆIM fill_ratio pobjeđuje

        Ovo je najjednostavniji i najdirektiniji pristup.
        """
        logger.info(f"[OMR PRECISE] Analiziram {len(options)} opcija sa preciznim koordinatama")

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
            # Izvuci koordinate
            x = int(option.get('x', 0))
            y = int(option.get('y', 0))
            opt_w = int(option.get('width', 40))
            opt_h = int(option.get('height', 40))

            # FULL RECTANGLE - koristi CELE dimenzije koje je korisnik namestio u editoru
            # Bez smanjivanja! Korisnik je već namestio pravougaonike da pokrivaju krugove.

            # Granice
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w, x + opt_w)
            y2 = min(h, y + opt_h)

            if x2 <= x1 or y2 <= y1:
                logger.warning(f"[OMR PRECISE] Opcija {option.get('label')} ima nevalidne koordinate: ({x1},{y1})-({x2},{y2})")
                continue

            # Crop ROI
            roi = binary[y1:y2, x1:x2]

            # Izračunaj fill_ratio
            white_pixels = cv2.countNonZero(roi)
            total_pixels = roi.shape[0] * roi.shape[1]
            fill_ratio = white_pixels / total_pixels if total_pixels > 0 else 0

            # Dodatna analiza: da li postoji kružna struktura?
            # Traži kontinuitet na ivicama (zaokruživanje ima liniju oko)
            edge_score = self._analyze_circle_edges(roi)

            # Ukupni score = fill_ratio * 0.6 + edge_score * 0.4
            combined_score = fill_ratio * 0.6 + edge_score * 0.4

            results.append({
                'label': option.get('label', str(i+1)),
                'value': option.get('value', option.get('label', str(i+1))),
                'fill_ratio': fill_ratio,
                'edge_score': edge_score,
                'combined_score': combined_score,
                'roi_size': (opt_w, opt_h)
            })

            logger.info(f"[OMR PRECISE] {option.get('label')}: fill={fill_ratio:.3f}, edge={edge_score:.3f}, combined={combined_score:.3f}")

        if not results:
            return {
                'value': None,
                'confidence': 0.0,
                'warning': 'Nema validnih opcija',
                'method': 'precise_coords'
            }

        # Nađi opciju sa NAJVEĆIM combined_score
        best = max(results, key=lambda x: x['combined_score'])

        # Provera: da li ima dovoljno "mastila"?
        MIN_SCORE_THRESHOLD = 0.10  # Povećano sa 0.08 da izbegnemo false positives
        if best['combined_score'] < MIN_SCORE_THRESHOLD:
            logger.warning(f"[OMR PRECISE] Najbolji score ({best['combined_score']:.3f}) je premali - verovatno nije označeno")
            return {
                'value': None,
                'confidence': 0.0,
                'warning': 'Nijedna opcija nije dovoljno označena',
                'method': 'precise_coords',
                'all_options': results
            }

        # Confidence: Koliko je bolji od drugog najboljeg?
        sorted_results = sorted(results, key=lambda x: x['combined_score'], reverse=True)

        if len(sorted_results) > 1:
            second_best = sorted_results[1]['combined_score']
            diff = best['combined_score'] - second_best

            # KRITIČNO: Ako je razlika mala (< 0.05), možda je greška
            if diff < 0.05:
                logger.warning(f"[OMR PRECISE] Razlika između 1. i 2. opcije je mala ({diff:.3f}) - možda je nejasno")
                confidence = 0.6  # Niži confidence zbog nejasnoće
            elif second_best > 0:
                # Razlika između prvog i drugog
                ratio = best['combined_score'] / (second_best + 0.001)
                # Ako je 2x bolji od drugog, confidence = 1.0
                confidence = min((ratio - 1.0) * 0.5 + 0.7, 1.0)
            else:
                confidence = 1.0
        else:
            confidence = 0.9

        # Bonus confidence ako ima dobar edge score (znači da je zaokruženo)
        if best['edge_score'] > 0.15:
            confidence = min(confidence + 0.1, 1.0)

        logger.info(f"[OMR PRECISE] ✓ Detekcija: {best['label']} = {best['value']} (conf={confidence:.2f})")

        return {
            'value': best['value'],
            'label': best['label'],
            'confidence': confidence,
            'fill_ratio': best['fill_ratio'],
            'edge_score': best['edge_score'],
            'combined_score': best['combined_score'],
            'method': 'precise_coords',
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

    def _detect_circles_by_contour(self, gray: np.ndarray, options: List[Dict]) -> Dict:
        """
        Pronalazi ručno nacrtane krugove analizom kontura.

        Ručno nacrtan krug ima karakteristike:
        - Zatvorena ili skoro zatvorena kontura
        - Aspect ratio blizu 1 (približno kružna)
        - Dovoljno velika površina
        - Nalazi se oko jedne od opcija
        """
        h, w = gray.shape[:2]

        # Preprocessing za bolju detekciju linija
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Canny edge detection - pronalazi ivice
        edges = cv2.Canny(blurred, 30, 100)

        # Dilate da spojimo prekide u ručno nacrtanim linijama
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)

        # Pronađi konture
        contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filtriraj konture koje izgledaju kao krugovi
        circle_candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Preskoči premale ili prevelike konture
            if area < self.min_contour_area or area > self.max_contour_area:
                continue

            # Izračunaj "kružnost" (circularity)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)

            # Ručno nacrtani krugovi imaju circularity ~0.4-0.9
            if circularity < self.circularity_threshold:
                continue

            # Proveri aspect ratio bounding box-a
            x, y, bw, bh = cv2.boundingRect(contour)
            aspect_ratio = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0

            # Krugovi imaju aspect ratio blizu 1
            if aspect_ratio < 0.4:
                continue

            # Ovo je kandidat za krug
            center_x = x + bw // 2
            center_y = y + bh // 2

            circle_candidates.append({
                'contour': contour,
                'center': (center_x, center_y),
                'bbox': (x, y, bw, bh),
                'area': area,
                'circularity': circularity,
                'aspect_ratio': aspect_ratio
            })

            logger.debug(f"Circle candidate: center=({center_x},{center_y}), "
                        f"area={area}, circ={circularity:.2f}, ar={aspect_ratio:.2f}")

        if not circle_candidates:
            return {
                'value': None,
                'confidence': 0.0,
                'warning': 'Nije pronađen nijedan krug',
                'method': 'contour_circle'
            }

        # Za svaku opciju, proveri da li ima krug oko sebe
        option_scores = []

        for option in options:
            opt_y = option.get('relative_y', option.get('y', 0))
            opt_h = option.get('height', h // len(options))
            opt_center_y = opt_y + opt_h // 2

            best_circle_score = 0

            for circle in circle_candidates:
                cx, cy = circle['center']
                _, _, bw, bh = circle['bbox']

                # Da li je centar kruga blizu centra opcije?
                y_distance = abs(cy - opt_center_y)

                # Krug treba biti u vertikalnoj zoni opcije
                if y_distance < opt_h * 0.8:
                    # Score baziran na blizini i kvalitetu kruga
                    proximity_score = 1.0 - (y_distance / (opt_h * 0.8))
                    quality_score = circle['circularity'] * circle['aspect_ratio']

                    total_score = proximity_score * quality_score * (circle['area'] / 1000)

                    if total_score > best_circle_score:
                        best_circle_score = total_score

            option_scores.append({
                'label': option['label'],
                'value': option.get('value', option['label']),
                'score': best_circle_score
            })

        # Pronađi opciju sa najboljim score-om
        if not option_scores:
            return {
                'value': None,
                'confidence': 0.0,
                'warning': 'Nema opcija za analizu',
                'method': 'contour_circle'
            }

        best = max(option_scores, key=lambda x: x['score'])

        if best['score'] < 0.01:
            return {
                'value': None,
                'confidence': 0.0,
                'warning': 'Krug nije dovoljno blizu nijedne opcije',
                'method': 'contour_circle',
                'all_options': option_scores
            }

        # Izračunaj confidence
        scores = [o['score'] for o in option_scores]
        max_score = best['score']
        other_scores = [s for s in scores if s != max_score]

        if other_scores and max(other_scores) > 0:
            confidence = min(max_score / (max(other_scores) + 0.01), 1.0)
        else:
            confidence = 0.8 if max_score > 0.1 else 0.5

        return {
            'value': best['value'],
            'label': best['label'],
            'confidence': confidence,
            'method': 'contour_circle',
            'all_options': option_scores
        }

    def _detect_by_edge_density(self, gray: np.ndarray, options: List[Dict]) -> Dict:
        """
        Detektuje zaokruženu opciju analizom gustine ivica (edges).

        Zaokružen broj ima više ivica oko sebe (od olovke/hemijske).
        """
        h, w = gray.shape[:2]

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Dilate edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        edges = cv2.dilate(edges, kernel, iterations=1)

        results = []

        for i, option in enumerate(options):
            opt_y = option.get('relative_y', option.get('y', 0))
            opt_h = option.get('height', h // len(options))

            # Osiguraj granice
            y1 = max(0, int(opt_y))
            y2 = min(h, int(opt_y + opt_h))

            if y2 <= y1:
                continue

            roi = edges[y1:y2, :]

            # Gustina ivica
            edge_density = cv2.countNonZero(roi) / roi.size if roi.size > 0 else 0

            # Analiziraj distribuciju ivica - zaokruženje ima ivice na periferiji
            roi_h, roi_w = roi.shape[:2]

            # Podeli ROI na centar i periferiju
            margin_h = roi_h // 4
            margin_w = roi_w // 4

            center_roi = roi[margin_h:roi_h-margin_h, margin_w:roi_w-margin_w]

            # Periferija
            top = roi[0:margin_h, :]
            bottom = roi[roi_h-margin_h:, :]
            left = roi[:, 0:margin_w]
            right = roi[:, roi_w-margin_w:]

            center_density = cv2.countNonZero(center_roi) / center_roi.size if center_roi.size > 0 else 0

            periphery_pixels = (cv2.countNonZero(top) + cv2.countNonZero(bottom) +
                               cv2.countNonZero(left) + cv2.countNonZero(right))
            periphery_size = top.size + bottom.size + left.size + right.size
            periphery_density = periphery_pixels / periphery_size if periphery_size > 0 else 0

            # Zaokruženje: više ivica na periferiji nego u centru
            # Score = periphery_density - center_density + total edge_density
            score = periphery_density * 2 + edge_density - center_density * 0.5

            results.append({
                'label': option['label'],
                'value': option.get('value', option['label']),
                'edge_density': edge_density,
                'periphery_density': periphery_density,
                'center_density': center_density,
                'score': score
            })

            logger.debug(f"Edge analysis {option['label']}: edge={edge_density:.4f}, "
                        f"periph={periphery_density:.4f}, center={center_density:.4f}, score={score:.4f}")

        if not results:
            return {'value': None, 'confidence': 0.0, 'warning': 'Nema rezultata', 'method': 'edge_density'}

        best = max(results, key=lambda x: x['score'])

        # Confidence
        scores = [r['score'] for r in results]
        max_score = best['score']
        other_scores = [s for s in scores if s != max_score]
        avg_other = sum(other_scores) / len(other_scores) if other_scores else 0

        if max_score < 0.02:
            return {
                'value': None,
                'confidence': 0.0,
                'warning': 'Premala gustina ivica',
                'method': 'edge_density',
                'all_options': results
            }

        if avg_other > 0:
            confidence = min((max_score / avg_other - 1) * 0.5 + 0.5, 1.0)
        else:
            confidence = 0.7

        return {
            'value': best['value'],
            'label': best['label'],
            'confidence': confidence,
            'method': 'edge_density',
            'all_options': results
        }

    def _detect_by_zone_density(self, gray: np.ndarray, options: List[Dict]) -> Dict:
        """
        Fallback: Zone-based ink density.
        Deli ROI na zone i meri ukupnu količinu "mastila" u svakoj.
        """
        h, w = gray.shape[:2]

        # Binarizacija
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 5
        )

        # Morfološko čišćenje
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        results = []

        for i, option in enumerate(options):
            opt_y = option.get('relative_y', option.get('y', 0))
            opt_h = option.get('height', h // len(options))

            y1 = max(0, int(opt_y))
            y2 = min(h, int(opt_y + opt_h))

            if y2 <= y1:
                continue

            roi = binary[y1:y2, :]

            # Ukupna gustina
            total_density = cv2.countNonZero(roi) / roi.size if roi.size > 0 else 0

            # Ring analysis (kao pre, ali pojednostavljeno)
            roi_h, roi_w = roi.shape[:2]
            center_x, center_y = roi_w // 2, roi_h // 2

            inner_r = min(roi_w, roi_h) // 4
            outer_r = min(roi_w, roi_h) // 2

            mask_outer = np.zeros_like(roi)
            mask_inner = np.zeros_like(roi)
            cv2.circle(mask_outer, (center_x, center_y), outer_r, 255, -1)
            cv2.circle(mask_inner, (center_x, center_y), inner_r, 255, -1)
            mask_ring = cv2.subtract(mask_outer, mask_inner)

            ring_ink = cv2.countNonZero(cv2.bitwise_and(roi, mask_ring))
            ring_area = cv2.countNonZero(mask_ring)
            ring_density = ring_ink / ring_area if ring_area > 0 else 0

            score = ring_density * 1.5 + total_density

            results.append({
                'label': option['label'],
                'value': option.get('value', option['label']),
                'total_density': total_density,
                'ring_density': ring_density,
                'score': score
            })

        if not results:
            return {'value': None, 'confidence': 0.0, 'warning': 'Nema rezultata', 'method': 'zone_density'}

        best = max(results, key=lambda x: x['score'])

        scores = [r['score'] for r in results]
        max_score = best['score']
        other_scores = [s for s in scores if s != max_score]
        avg_other = sum(other_scores) / len(other_scores) if other_scores else 0

        if max_score < 0.01:
            return {
                'value': None,
                'confidence': 0.0,
                'warning': 'Nijedna opcija nije označena',
                'method': 'zone_density',
                'all_options': results
            }

        if avg_other > 0:
            confidence = min((max_score / avg_other - 1) * 0.4 + 0.4, 0.9)
        else:
            confidence = 0.6

        return {
            'value': best['value'],
            'label': best['label'],
            'confidence': confidence,
            'method': 'zone_density',
            'all_options': results
        }

    def _prepare_options(self, options: List[Dict], img_h: int, img_w: int) -> List[Dict]:
        """Pripremi opcije - dodaj koordinate ako ih nema."""
        prepared = []

        # Ako opcije nemaju koordinate, rasporedi ih vertikalno
        if options and 'y' not in options[0] and 'relative_y' not in options[0]:
            row_height = img_h // len(options) if len(options) > 0 else img_h
            for i, opt in enumerate(options):
                new_opt = opt.copy()
                new_opt['relative_y'] = i * row_height
                new_opt['height'] = row_height
                new_opt['width'] = img_w
                prepared.append(new_opt)
        else:
            prepared = [opt.copy() for opt in options]

        return prepared

    def detect_checkbox(self,
                        image: np.ndarray,
                        checkbox_coords: Dict = None) -> Tuple[bool, float]:
        """Detektuje da li je checkbox označen."""
        gray = self._to_grayscale(image)

        if checkbox_coords:
            roi = self._crop_roi(gray, checkbox_coords)
        else:
            roi = gray

        binary = cv2.adaptiveThreshold(
            roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        fill_ratio = cv2.countNonZero(binary) / binary.size if binary.size > 0 else 0

        is_checked = fill_ratio >= self.fill_threshold
        confidence = min(fill_ratio / self.fill_threshold, 1.0) if is_checked else 1.0 - fill_ratio

        return is_checked, float(confidence)

    def detect_multi_select(self,
                            image: np.ndarray,
                            options: List[Dict]) -> Dict:
        """Detektuje više označenih opcija."""
        gray = self._to_grayscale(image)
        h, w = gray.shape[:2]

        # NOVA LOGIKA: Ako opcije imaju precizne koordinate, koristi fill_ratio
        has_precise_coords = all('x' in opt and 'y' in opt for opt in options)

        if has_precise_coords:
            logger.info("[OMR MULTI] Koristim precizne koordinate za multi-select detekciju")
            return self._detect_multi_by_precise_coords(gray, options)

        # STARA LOGIKA: Fallback na edge-based
        options = self._prepare_options(options, h, w)

        # Edge-based analysis za svaku opciju
        edges = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        edges = cv2.dilate(edges, kernel, iterations=1)

        detected_values = []
        results = []

        # Izračunaj prosečnu gustinu
        avg_density = cv2.countNonZero(edges) / edges.size if edges.size > 0 else 0
        threshold = avg_density * 1.5

        for option in options:
            opt_y = option.get('relative_y', option.get('y', 0))
            opt_h = option.get('height', h // len(options))

            y1 = max(0, int(opt_y))
            y2 = min(h, int(opt_y + opt_h))

            if y2 <= y1:
                continue

            roi = edges[y1:y2, :]
            density = cv2.countNonZero(roi) / roi.size if roi.size > 0 else 0

            is_detected = density > threshold

            results.append({
                'label': option['label'],
                'value': option.get('value', option['label']),
                'density': density,
                'detected': is_detected
            })

            if is_detected:
                detected_values.append(option.get('value', option['label']))

        return {
            'values': detected_values,
            'count': len(detected_values),
            'confidence': 0.8 if detected_values else 0.0,
            'method': 'multi_select_edge',
            'all_options': results
        }

    def _detect_multi_by_precise_coords(self, gray: np.ndarray, options: List[Dict]) -> Dict:
        """
        Multi-select sa preciznim koordinatama.
        Za svaki krug, proveri fill_ratio - ako je > threshold, označen je.
        """
        logger.info(f"[OMR MULTI PRECISE] Analiziram {len(options)} opcija")

        # Binarizacija
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        binary = cv2.GaussianBlur(binary, (3, 3), 0)
        _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)

        h, w = gray.shape[:2]
        detected_values = []
        results = []

        # Threshold za multi-select - POVEĆAN da izbegnemo false positives
        # Threshold 0.15 znači da opcija mora imati bar 15% "mastila" da bi bila detektovana
        FILL_THRESHOLD = 0.15

        for i, option in enumerate(options):
            x = int(option.get('x', 0))
            y = int(option.get('y', 0))
            opt_w = int(option.get('width', 40))
            opt_h = int(option.get('height', 40))

            # FULL RECTANGLE - koristi CELE dimenzije koje je korisnik namestio
            # Bez smanjivanja!

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w, x + opt_w)
            y2 = min(h, y + opt_h)

            if x2 <= x1 or y2 <= y1:
                logger.warning(f"[OMR MULTI PRECISE] Opcija {option.get('label')} ima nevalidne koordinate nakon margin")
                continue

            roi = binary[y1:y2, x1:x2]

            white_pixels = cv2.countNonZero(roi)
            total_pixels = roi.shape[0] * roi.shape[1]
            fill_ratio = white_pixels / total_pixels if total_pixels > 0 else 0

            edge_score = self._analyze_circle_edges(roi)
            combined_score = fill_ratio * 0.6 + edge_score * 0.4

            is_detected = combined_score >= FILL_THRESHOLD

            results.append({
                'label': option.get('label', str(i+1)),
                'value': option.get('value', option.get('label', str(i+1))),
                'fill_ratio': fill_ratio,
                'edge_score': edge_score,
                'combined_score': combined_score,
                'detected': is_detected
            })

            logger.info(f"[OMR MULTI PRECISE] {option.get('label')}: fill={fill_ratio:.3f}, edge={edge_score:.3f}, combined={combined_score:.3f} → {'✓' if is_detected else '✗'}")

            if is_detected:
                detected_values.append(option.get('value', option.get('label', str(i+1))))

        confidence = 0.9 if detected_values else 0.0

        logger.info(f"[OMR MULTI PRECISE] ✓ Detektovano {len(detected_values)} opcija: {detected_values}")

        return {
            'values': detected_values,
            'count': len(detected_values),
            'confidence': confidence,
            'method': 'multi_select_precise',
            'all_options': results
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


if __name__ == "__main__":
    # Test
    print("Testing OMR Processor...")
    processor = OMRProcessor()

    # Kreiraj test sliku
    test_img = np.ones((200, 400), dtype=np.uint8) * 255

    # 4 opcije, treća je zaokružena
    for i, num in enumerate(['1', '2', '3', '4']):
        y = 20 + i * 45
        cv2.putText(test_img, num, (180, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Nacrtaj krug oko "3"
    cv2.ellipse(test_img, (190, 135), (30, 20), 0, 0, 360, (0, 0, 0), 2)

    options = [
        {'label': '1', 'value': 'Opcija 1'},
        {'label': '2', 'value': 'Opcija 2'},
        {'label': '3', 'value': 'Opcija 3'},
        {'label': '4', 'value': 'Opcija 4'},
    ]

    result = processor.detect_marked_option(test_img, options)
    print(f"Detected: {result.get('value')} (confidence: {result.get('confidence', 0):.2f})")
    print(f"Method: {result.get('method')}")

    expected = 'Opcija 3'
    print(f"\nTest {'PASSED' if result.get('value') == expected else 'FAILED'}")
