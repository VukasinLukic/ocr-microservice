"""
SV-20 OCR Mikroservis - OMR Logic
Optical Mark Recognition - detekcija zaokruzenih/oznacenih opcija.
Koristi se za polja kao sto su: Pol, Vrsta studija, Semestar, itd.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class OMRProcessor:
    """
    Optical Mark Recognition - detekcija zaokruzenih/oznacenih opcija.
    Koristi se za polja kao sto su: Pol, Godina studija, Semestar.
    """

    def __init__(self,
                 fill_threshold: float = 0.25,
                 min_contour_area: int = 50,
                 circle_detection_threshold: float = 0.3):
        """
        Args:
            fill_threshold: Procenat popunjenosti za detekciju (0.0-1.0)
            min_contour_area: Minimalna povrsina konture
            circle_detection_threshold: Prag za detekciju kruga
        """
        self.fill_threshold = fill_threshold
        self.min_contour_area = min_contour_area
        self.circle_detection_threshold = circle_detection_threshold

    def detect_marked_option(self,
                             image: np.ndarray,
                             options: List[Dict],
                             method: str = "fill_ratio") -> Dict:
        """
        Detektuje koja opcija je oznacena/zaokruzena.

        Args:
            image: Slika polja sa opcijama
            options: Lista opcija sa koordinatama iz template-a
                    [{"label": "1", "value": "M", "relative_y": 20}, ...]
            method: "fill_ratio" ili "circle_detection"

        Returns:
            Dict sa detektovanom opcijom i confidence
        """
        if method == "circle_detection":
            return self.detect_circled_option(image, options)

        gray = self._to_grayscale(image)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        h, w = binary.shape[:2]
        results = []

        for option in options:
            # Izracunaj ROI za ovu opciju
            # Opcije mogu imati relative_y ili apsolutne koordinate
            if 'x' in option and 'y' in option:
                # Apsolutne koordinate
                x = option.get('x', 0)
                y = option.get('y', 0)
                opt_w = option.get('width', 50)
                opt_h = option.get('height', 30)
            else:
                # Relativne koordinate - opcija se odnosi na redove u polju
                x = 0
                y = option.get('relative_y', 0)
                opt_w = w
                opt_h = 25  # Default visina reda

            # Osiguraj da je ROI validan
            x = max(0, min(x, w))
            y = max(0, min(y, h))
            x2 = min(w, x + opt_w)
            y2 = min(h, y + opt_h)

            if x2 <= x or y2 <= y:
                continue

            roi = binary[y:y2, x:x2]

            if roi.size == 0:
                continue

            fill_ratio = self._calculate_fill_ratio(roi)

            results.append({
                'label': option['label'],
                'value': option.get('value', option['label']),
                'fill_ratio': fill_ratio,
                'detected': fill_ratio >= self.fill_threshold
            })

            logger.debug(f"Opcija {option['label']}: fill_ratio={fill_ratio:.3f}")

        # Odredi rezultat
        detected_options = [r for r in results if r['detected']]

        if len(detected_options) == 1:
            return {
                'value': detected_options[0]['value'],
                'label': detected_options[0]['label'],
                'confidence': min(detected_options[0]['fill_ratio'] / self.fill_threshold * 0.5 + 0.5, 1.0),
                'method': 'fill_ratio',
                'all_options': results
            }
        elif len(detected_options) > 1:
            # Vise opcija detektovano - uzmi onu sa najvecim fill_ratio
            best = max(detected_options, key=lambda x: x['fill_ratio'])
            return {
                'value': best['value'],
                'label': best['label'],
                'confidence': 0.6,  # Nizi confidence zbog visesmislenosti
                'warning': 'Vise opcija detektovano',
                'method': 'fill_ratio',
                'all_options': results
            }
        else:
            # Nijedna opcija nije detektovana - vrati onu sa najvecim fill_ratio
            if results:
                best = max(results, key=lambda x: x['fill_ratio'])
                return {
                    'value': best['value'],
                    'label': best['label'],
                    'confidence': best['fill_ratio'],
                    'warning': 'Nijedna opcija nije jasno detektovana',
                    'method': 'fill_ratio',
                    'all_options': results
                }
            return {
                'value': None,
                'label': None,
                'confidence': 0.0,
                'warning': 'Nije moguce analizirati opcije',
                'method': 'fill_ratio',
                'all_options': []
            }

    def detect_circled_option(self,
                              image: np.ndarray,
                              options: List[Dict]) -> Dict:
        """
        Detektuje tekst koji je zaokruzen olovkom.
        Koristi Hough Circle Transform i detekciju kontura elipsi.
        """
        gray = self._to_grayscale(image)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Pronadji konture
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Pokusaj pronaci krugove/elipse
        circles = []
        for contour in contours:
            if len(contour) >= 5:
                try:
                    ellipse = cv2.fitEllipse(contour)
                    (cx, cy), (ma, MA), angle = ellipse

                    # Proveri da li lici na krug (odnos osa blizu 1)
                    if ma > 0 and MA > 0:
                        ratio = max(ma, MA) / min(ma, MA)
                        if ratio < 2.5:  # Dopusti blago elipticne oblike
                            circles.append({
                                'center': (int(cx), int(cy)),
                                'axes': (int(ma / 2), int(MA / 2)),
                                'area': np.pi * ma * MA / 4,
                                'contour': contour
                            })
                except cv2.error:
                    continue

        # Hough Circle Transform kao alternativa
        circles_hough = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=50
        )

        if circles_hough is not None:
            for circle in circles_hough[0]:
                x, y, r = circle
                circles.append({
                    'center': (int(x), int(y)),
                    'axes': (int(r), int(r)),
                    'area': np.pi * r * r,
                    'contour': None
                })

        # Proveri koja opcija ima krug oko sebe
        h, w = image.shape[:2]
        for option in options:
            if 'x' in option and 'y' in option:
                opt_center = (
                    option['x'] + option.get('width', 40) // 2,
                    option['y'] + option.get('height', 30) // 2
                )
            else:
                # Proceni centar opcije
                opt_center = (w // 2, option.get('relative_y', 0) + 15)

            for circle in circles:
                dist = np.sqrt(
                    (circle['center'][0] - opt_center[0]) ** 2 +
                    (circle['center'][1] - opt_center[1]) ** 2
                )

                # Ako je centar kruga blizu centra opcije
                threshold = max(option.get('width', 50), option.get('height', 30))
                if dist < threshold:
                    return {
                        'value': option.get('value', option['label']),
                        'label': option['label'],
                        'confidence': 0.85,
                        'method': 'circle_detection'
                    }

        # Fallback na fill_ratio metod
        logger.debug("Circle detection nije pronasao krugove, koristim fill_ratio")
        return self.detect_marked_option(image, options, method="fill_ratio")

    def detect_checkbox(self,
                        image: np.ndarray,
                        checkbox_coords: Dict = None) -> Tuple[bool, float]:
        """
        Detektuje da li je checkbox oznacen (X, check mark, popunjen).

        Args:
            image: Slika checkbox-a
            checkbox_coords: Opcione koordinate {x, y, width, height}

        Returns:
            (is_checked, confidence)
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

        # Pronadji konture unutar checkbox-a
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Proveri da li ima znacajnu oznaku unutra
        has_mark = False
        total_mark_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_contour_area:
                total_mark_area += area
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h) if h > 0 else 0

                # X ili check mark ima razuman aspect ratio
                if 0.3 < aspect_ratio < 3.0:
                    has_mark = True

        # Izracunaj fill ratio
        fill_ratio = self._calculate_fill_ratio(binary)

        # Kombinuj informacije
        is_checked = has_mark or fill_ratio >= self.fill_threshold

        if is_checked:
            confidence = min(fill_ratio / self.fill_threshold, 1.0) * 0.7 + 0.3
        else:
            confidence = 1.0 - fill_ratio

        return is_checked, float(confidence)

    def detect_multi_select(self,
                            image: np.ndarray,
                            options: List[Dict]) -> Dict:
        """
        Detektuje vise oznacenih opcija (za pitanja sa visestrukim izborom).

        Args:
            image: Slika polja
            options: Lista opcija

        Returns:
            Dict sa listom detektovanih opcija
        """
        gray = self._to_grayscale(image)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        h, w = binary.shape[:2]
        detected_values = []
        results = []

        for option in options:
            if 'x' in option and 'y' in option:
                x = option.get('x', 0)
                y = option.get('y', 0)
                opt_w = option.get('width', 50)
                opt_h = option.get('height', 30)
            else:
                x = 0
                y = option.get('relative_y', 0)
                opt_w = w
                opt_h = 25

            x = max(0, min(x, w))
            y = max(0, min(y, h))
            x2 = min(w, x + opt_w)
            y2 = min(h, y + opt_h)

            if x2 <= x or y2 <= y:
                continue

            roi = binary[y:y2, x:x2]
            if roi.size == 0:
                continue

            fill_ratio = self._calculate_fill_ratio(roi)
            is_detected = fill_ratio >= self.fill_threshold

            results.append({
                'label': option['label'],
                'value': option.get('value', option['label']),
                'fill_ratio': fill_ratio,
                'detected': is_detected
            })

            if is_detected:
                detected_values.append(option.get('value', option['label']))

        return {
            'values': detected_values,
            'count': len(detected_values),
            'confidence': 0.8 if detected_values else 0.0,
            'method': 'multi_select',
            'all_options': results
        }

    def analyze_number_selection(self,
                                 image: np.ndarray,
                                 number_range: Tuple[int, int] = (1, 9)) -> Dict:
        """
        Analizira selekciju broja (npr. 1-9 za vrstu studija).
        Koristi kombinaciju detekcije kruga i fill ratio.

        Args:
            image: Slika sa brojevima
            number_range: Opseg brojeva (min, max)

        Returns:
            Dict sa detektovanim brojem
        """
        min_num, max_num = number_range
        options = []

        # Generisi opcije za svaki broj
        h, w = image.shape[:2] if len(image.shape) == 2 else image.shape[:2]
        num_options = max_num - min_num + 1
        row_height = h // num_options if num_options > 0 else h

        for i, num in enumerate(range(min_num, max_num + 1)):
            options.append({
                'label': str(num),
                'value': str(num),
                'relative_y': i * row_height,
                'height': row_height
            })

        return self.detect_marked_option(image, options)

    def _to_grayscale(self, img: np.ndarray) -> np.ndarray:
        """Konvertuje u grayscale ako nije vec."""
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

    def _calculate_fill_ratio(self, binary_image: np.ndarray) -> float:
        """Racuna procenat belih piksela u binarnoj slici."""
        if binary_image.size == 0:
            return 0.0

        white_pixels = cv2.countNonZero(binary_image)
        total_pixels = binary_image.shape[0] * binary_image.shape[1]

        return white_pixels / total_pixels if total_pixels > 0 else 0.0


def test_omr_processor():
    """Test funkcija za OMR processor."""
    print("Testiram OMR Processor...")

    processor = OMRProcessor(fill_threshold=0.25)

    # Kreiraj test sliku
    test_img = np.ones((200, 100), dtype=np.uint8) * 255

    # Simuliraj zaokruzenu opciju na poziciji y=50
    cv2.circle(test_img, (50, 60), 20, (0, 0, 0), 2)
    # Dodaj malo crne unutar kruga
    cv2.circle(test_img, (50, 60), 15, (100, 100, 100), -1)

    options = [
        {'label': '1', 'value': 'Option1', 'relative_y': 10, 'height': 40},
        {'label': '2', 'value': 'Option2', 'relative_y': 50, 'height': 40},
        {'label': '3', 'value': 'Option3', 'relative_y': 90, 'height': 40},
    ]

    result = processor.detect_marked_option(test_img, options)
    print(f"Detektovana opcija: {result['value']} (confidence: {result['confidence']:.2f})")
    print(f"Sve opcije: {result['all_options']}")

    print("Test zavrsen!")


if __name__ == "__main__":
    test_omr_processor()
