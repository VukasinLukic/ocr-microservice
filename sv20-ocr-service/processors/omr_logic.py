"""
SV-20 OCR Mikroservis - OMR Logic (Improved)
Optical Mark Recognition - detekcija ZAOKRUŽENIH BROJEVA.

SV-20 obrazac koristi zaokruživanje brojeva (circled numbers),
a ne popunjavanje kružića (bubbles). Ova verzija je optimizovana
za detekciju ručno nacrtanih krugova oko štampanih brojeva.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class OMRProcessor:
    """
    Optical Mark Recognition - detekcija zaokruženih brojeva.

    SV-20 obrazac koristi stil gde student zaokružuje štampani broj,
    umesto da popunjava kružić. Ova klasa koristi kombinaciju:
    1. Detekcija kontura (zaokruženih oblasti)
    2. Analiza "ink density" oko svakog broja
    3. Hough Circle Transform za dodatnu verifikaciju
    """

    def __init__(self,
                 fill_threshold: float = 0.15,
                 min_contour_area: int = 100,
                 circle_detection_threshold: float = 0.3):
        """
        Args:
            fill_threshold: Procenat popunjenosti za detekciju (0.0-1.0)
            min_contour_area: Minimalna površina konture
            circle_detection_threshold: Prag za detekciju kruga
        """
        self.fill_threshold = fill_threshold
        self.min_contour_area = min_contour_area
        self.circle_detection_threshold = circle_detection_threshold

    def detect_marked_option(self,
                             image: np.ndarray,
                             options: List[Dict],
                             method: str = "auto") -> Dict:
        """
        Glavna metoda za detekciju označene opcije.

        Koristi kombinaciju metoda:
        1. Prvo pokušava detekciju zaokruženih brojeva (circle ink detection)
        2. Fallback na pixel density ako krug nije detektovan

        Args:
            image: Slika ROI-a sa opcijama
            options: Lista opcija sa koordinatama
            method: "auto", "circle", ili "density"

        Returns:
            Dict sa detektovanom opcijom, confidence, itd.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape[:2]

        if not options:
            return {'value': None, 'confidence': 0.0, 'warning': 'Nema definisanih opcija'}

        # Pripremi opcije ako nemaju koordinate
        options = self._prepare_options(options, h, w)

        # METODA 1: Detekcija zaokruženih oblasti (ink around numbers)
        circle_result = self._detect_circled_number(gray, options)

        if circle_result['confidence'] > 0.5:
            logger.debug(f"Circle detection uspešno: {circle_result['value']} (conf={circle_result['confidence']:.2f})")
            return circle_result

        # METODA 2: Pixel density (fallback)
        density_result = self._detect_by_ink_density(gray, options)

        # Odaberi bolji rezultat
        if circle_result['confidence'] > density_result['confidence']:
            return circle_result
        return density_result

    def _detect_circled_number(self, gray: np.ndarray, options: List[Dict]) -> Dict:
        """
        Detektuje koji broj je zaokružen analizom "mastila" oko svakog broja.

        Logika: Zaokružen broj ima više mastila (ink) oko sebe nego nezaokruženi.
        Merimo količinu crnih piksela u PRSTENU oko centra svake opcije.
        """
        h, w = gray.shape[:2]

        # Binarizacija
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 5
        )

        # Ukloni sitne šumove
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        results = []

        for option in options:
            # Dobij ROI za ovu opciju
            x = option.get('x', 0)
            y = option.get('relative_y', option.get('y', 0))
            opt_w = option.get('width', w)
            opt_h = option.get('height', 30)

            # Osiguraj granice
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            x2 = min(w, x + opt_w)
            y2 = min(h, y + opt_h)

            if x2 <= x or y2 <= y:
                results.append({
                    'label': option['label'],
                    'value': option.get('value', option['label']),
                    'ring_ink': 0,
                    'total_ink': 0,
                    'score': 0
                })
                continue

            roi = binary[y:y2, x:x2]
            roi_h, roi_w = roi.shape[:2]

            if roi_h < 5 or roi_w < 5:
                continue

            # Izračunaj "ink" u različitim zonama
            # Centar (gde je štampani broj) vs. prsten oko centra (gde je zaokruženje)

            center_x, center_y = roi_w // 2, roi_h // 2

            # Kreiraj masku za PRSTEN (ring) oko centra
            # Unutrašnji krug (štampani broj) - ignorišemo
            # Spoljašnji prsten (zaokruženje) - merimo

            inner_radius = min(roi_w, roi_h) // 4  # Unutrašnji radius
            outer_radius = min(roi_w, roi_h) // 2  # Spoljašnji radius

            mask_outer = np.zeros_like(roi)
            mask_inner = np.zeros_like(roi)

            cv2.circle(mask_outer, (center_x, center_y), outer_radius, 255, -1)
            cv2.circle(mask_inner, (center_x, center_y), inner_radius, 255, -1)

            # Ring maska = outer - inner
            mask_ring = cv2.subtract(mask_outer, mask_inner)

            # Ink u prstenu (zaokruženje)
            ring_ink = cv2.countNonZero(cv2.bitwise_and(roi, mask_ring))
            ring_area = cv2.countNonZero(mask_ring)

            # Ukupan ink u ROI-u
            total_ink = cv2.countNonZero(roi)
            total_area = roi_h * roi_w

            # Score: odnos ink-a u prstenu prema ukupnom
            # Zaokružen broj ima više ink-a u prstenu
            ring_density = ring_ink / ring_area if ring_area > 0 else 0
            total_density = total_ink / total_area if total_area > 0 else 0

            # Score kombinuje ring density i total density
            # Zaokružen broj ima visoku ring density
            score = ring_density * 2 + total_density

            results.append({
                'label': option['label'],
                'value': option.get('value', option['label']),
                'ring_ink': ring_ink,
                'ring_density': ring_density,
                'total_ink': total_ink,
                'total_density': total_density,
                'score': score
            })

            logger.debug(f"Option {option['label']}: ring_density={ring_density:.4f}, "
                        f"total_density={total_density:.4f}, score={score:.4f}")

        if not results:
            return {
                'value': None,
                'confidence': 0.0,
                'warning': 'Nije moguće analizirati opcije'
            }

        # Pronađi opciju sa najboljim score-om
        best = max(results, key=lambda x: x['score'])

        # Izračunaj confidence
        scores = [r['score'] for r in results]
        max_score = best['score']

        if max_score < 0.01:
            return {
                'value': None,
                'confidence': 0.0,
                'warning': 'Nijedna opcija nije zaokružena',
                'all_options': results
            }

        # Confidence: koliko je best bolji od ostalih
        other_scores = [s for s in scores if s != max_score]
        avg_other = sum(other_scores) / len(other_scores) if other_scores else 0

        if avg_other > 0:
            confidence = min((max_score / avg_other - 1) * 0.5, 1.0)
        else:
            confidence = 0.8 if max_score > 0.05 else 0.3

        return {
            'value': best['value'],
            'label': best['label'],
            'confidence': confidence,
            'method': 'ring_ink_detection',
            'all_options': results
        }

    def _detect_by_ink_density(self, gray: np.ndarray, options: List[Dict]) -> Dict:
        """
        Fallback metoda: detektuje opciju sa najviše "mastila" (ink).
        """
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        h, w = binary.shape[:2]
        results = []

        for option in options:
            x = option.get('x', 0)
            y = option.get('relative_y', option.get('y', 0))
            opt_w = option.get('width', w)
            opt_h = option.get('height', 30)

            x = max(0, min(x, w))
            y = max(0, min(y, h))
            x2 = min(w, x + opt_w)
            y2 = min(h, y + opt_h)

            if x2 <= x or y2 <= y:
                continue

            roi = binary[y:y2, x:x2]

            # Morfološko čišćenje - ukloni tekst, zadrži deblje linije (zaokruženje)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            roi_cleaned = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel, iterations=1)

            density = cv2.countNonZero(roi_cleaned) / roi_cleaned.size if roi_cleaned.size > 0 else 0

            results.append({
                'label': option['label'],
                'value': option.get('value', option['label']),
                'density': density
            })

        if not results:
            return {'value': None, 'confidence': 0.0, 'warning': 'Nema rezultata'}

        best = max(results, key=lambda x: x['density'])
        max_density = best['density']

        other_densities = [r['density'] for r in results if r != best]
        avg_other = sum(other_densities) / len(other_densities) if other_densities else 0

        if max_density < 0.02:
            return {
                'value': None,
                'confidence': 0.0,
                'warning': 'Nijedna opcija nije označena',
                'all_options': results
            }

        confidence = 0.0
        if avg_other > 0:
            ratio = max_density / avg_other
            confidence = min((ratio - 1.0) * 0.5, 1.0)
        else:
            confidence = 0.7

        return {
            'value': best['value'],
            'label': best['label'],
            'confidence': confidence,
            'method': 'ink_density',
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
        """
        Detektuje da li je checkbox označen.
        """
        gray = self._to_grayscale(image)

        if checkbox_coords:
            roi = self._crop_option_roi(gray, checkbox_coords)
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
        """
        Detektuje više označenih opcija (za pitanja sa višestrukim izborom).
        Svaka opcija se proverava pojedinačno.
        """
        gray = self._to_grayscale(image)
        h, w = gray.shape[:2]

        options = self._prepare_options(options, h, w)

        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 5
        )

        detected_values = []
        results = []

        # Izračunaj prosečnu gustinu za određivanje praga
        total_density = cv2.countNonZero(binary) / binary.size if binary.size > 0 else 0
        dynamic_threshold = max(total_density * 1.5, 0.03)  # Minimalno 3%

        for option in options:
            x = option.get('x', 0)
            y = option.get('relative_y', option.get('y', 0))
            opt_w = option.get('width', w)
            opt_h = option.get('height', 30)

            x = max(0, min(x, w))
            y = max(0, min(y, h))
            x2 = min(w, x + opt_w)
            y2 = min(h, y + opt_h)

            if x2 <= x or y2 <= y:
                continue

            roi = binary[y:y2, x:x2]

            # Analiziraj "prsten" oko centra kao u detect_circled_number
            roi_h, roi_w = roi.shape[:2]
            center_x, center_y = roi_w // 2, roi_h // 2

            inner_radius = min(roi_w, roi_h) // 4
            outer_radius = min(roi_w, roi_h) // 2

            mask_outer = np.zeros_like(roi)
            mask_inner = np.zeros_like(roi)

            cv2.circle(mask_outer, (center_x, center_y), outer_radius, 255, -1)
            cv2.circle(mask_inner, (center_x, center_y), inner_radius, 255, -1)

            mask_ring = cv2.subtract(mask_outer, mask_inner)

            ring_ink = cv2.countNonZero(cv2.bitwise_and(roi, mask_ring))
            ring_area = cv2.countNonZero(mask_ring)
            ring_density = ring_ink / ring_area if ring_area > 0 else 0

            # Da li je zaokruženo?
            is_detected = ring_density > dynamic_threshold

            results.append({
                'label': option['label'],
                'value': option.get('value', option['label']),
                'ring_density': ring_density,
                'detected': is_detected
            })

            if is_detected:
                detected_values.append(option.get('value', option['label']))

            logger.debug(f"Multi-select {option['label']}: ring_density={ring_density:.4f}, "
                        f"threshold={dynamic_threshold:.4f}, detected={is_detected}")

        return {
            'values': detected_values,
            'count': len(detected_values),
            'confidence': 0.8 if detected_values else 0.0,
            'method': 'multi_select_ring',
            'all_options': results
        }

    def detect_circled_option_hough(self, image: np.ndarray, options: List[Dict]) -> Dict:
        """
        Alternativna metoda: koristi Hough Circle Transform.
        Bolji za jasno nacrtane krugove.
        """
        gray = self._to_grayscale(image)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Pokušaj pronaći krugove
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=20,
            param1=50,
            param2=25,
            minRadius=8,
            maxRadius=40
        )

        if circles is None:
            logger.debug("Hough Circle Transform nije pronašao krugove")
            return self._detect_circled_number(gray, options)

        h, w = gray.shape[:2]
        detected_circles = []

        for circle in circles[0]:
            cx, cy, r = circle
            detected_circles.append({
                'center': (int(cx), int(cy)),
                'radius': int(r)
            })

        # Proveri koja opcija ima krug
        for option in options:
            opt_x = option.get('x', 0)
            opt_y = option.get('relative_y', option.get('y', 0))
            opt_w = option.get('width', 50)
            opt_h = option.get('height', 30)

            opt_center = (opt_x + opt_w // 2, opt_y + opt_h // 2)

            for circle in detected_circles:
                dist = np.sqrt(
                    (circle['center'][0] - opt_center[0]) ** 2 +
                    (circle['center'][1] - opt_center[1]) ** 2
                )

                if dist < max(opt_w, opt_h):
                    return {
                        'value': option.get('value', option['label']),
                        'label': option['label'],
                        'confidence': 0.9,
                        'method': 'hough_circle'
                    }

        # Fallback
        return self._detect_circled_number(gray, options)

    def _to_grayscale(self, img: np.ndarray) -> np.ndarray:
        """Konvertuje u grayscale ako nije već."""
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _crop_option_roi(self, image: np.ndarray, coords: Dict) -> np.ndarray:
        """Iseca ROI na osnovu koordinata."""
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


def test_omr_processor():
    """Test funkcija za OMR processor."""
    print("Testiram poboljšani OMR Processor za zaokružene brojeve...")

    processor = OMRProcessor(fill_threshold=0.15)

    # Kreiraj test sliku - simuliraj zaokruženi broj
    test_img = np.ones((150, 400), dtype=np.uint8) * 255

    # Simuliraj 3 opcije, druga je zaokružena
    # Opcija 1 - nezaokružena
    cv2.putText(test_img, "1", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Opcija 2 - ZAOKRUŽENA
    cv2.putText(test_img, "2", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.ellipse(test_img, (55, 70), (25, 18), 0, 0, 360, (0, 0, 0), 2)  # Ručno nacrtan krug

    # Opcija 3 - nezaokružena
    cv2.putText(test_img, "3", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    options = [
        {'label': '1', 'value': 'Option1', 'relative_y': 0, 'height': 50, 'width': 100},
        {'label': '2', 'value': 'Option2', 'relative_y': 50, 'height': 50, 'width': 100},
        {'label': '3', 'value': 'Option3', 'relative_y': 100, 'height': 50, 'width': 100},
    ]

    result = processor.detect_marked_option(test_img, options)
    print(f"Detektovana opcija: {result.get('value')} (confidence: {result.get('confidence', 0):.2f})")
    print(f"Metoda: {result.get('method', 'N/A')}")

    if 'all_options' in result:
        print("\nSve opcije:")
        for opt in result['all_options']:
            print(f"  {opt['label']}: score={opt.get('score', opt.get('density', 0)):.4f}")

    # Očekujemo da detektuje opciju 2
    expected = 'Option2'
    actual = result.get('value')
    print(f"\nTest {'PASSED ✓' if actual == expected else 'FAILED ✗'}")
    print(f"  Očekivano: {expected}, Dobijeno: {actual}")


if __name__ == "__main__":
    test_omr_processor()
