"""
SV-20 OCR Mikroservis - Image Processor
OpenCV pipeline za obradu slike SV-20 obrasca.
Podrzava: deskew, denoise, ROI cropping, uklanjanje linija/kucica, perspective transform.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict
import logging

logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    OpenCV pipeline za obradu slike SV-20 obrasca.
    Podrzava: deskew, denoise, ROI cropping, uklanjanje linija/kucica.
    """

    def __init__(self, target_dpi: int = 300):
        self.target_dpi = target_dpi

    def load_image(self, image_path: str) -> np.ndarray:
        """Ucitava sliku sa diska."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Ne mogu ucitati sliku: {image_path}")
        logger.info(f"Ucitana slika: {image_path}, dimenzije: {img.shape}")
        return img

    def load_image_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Ucitava sliku iz bajtova (za upload)."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Ne mogu dekodirati sliku iz bajtova")
        logger.info(f"Ucitana slika iz bajtova, dimenzije: {img.shape}")
        return img

    def to_grayscale(self, img: np.ndarray) -> np.ndarray:
        """Konvertuje u grayscale ako nije vec."""
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def deskew(self, img: np.ndarray, max_angle: float = 10.0) -> np.ndarray:
        """
        Ispravlja nagib slike koristeci Hough Line Transform.

        Args:
            img: Ulazna slika
            max_angle: Maksimalan ugao korekcije (stepeni)

        Returns:
            Ispravljena slika
        """
        gray = self.to_grayscale(img)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100,
                                minLineLength=100, maxLineGap=10)

        if lines is None:
            logger.info("Deskew: nije pronadjena nijedna linija")
            return img

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 45:  # Samo horizontalne linije
                angles.append(angle)

        if not angles:
            return img

        median_angle = np.median(angles)

        # Ogranici ugao korekcije
        if abs(median_angle) > max_angle:
            median_angle = max_angle if median_angle > 0 else -max_angle

        if abs(median_angle) < 0.5:
            logger.info(f"Deskew: ugao {median_angle:.2f}° je prenizak, preskacemo")
            return img

        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)

        logger.info(f"Deskew: korigovano {median_angle:.2f} stepeni")
        return rotated

    def denoise(self, img: np.ndarray, strength: int = 10) -> np.ndarray:
        """
        Uklanja sum koristeci Non-local Means Denoising.

        Args:
            img: Ulazna slika
            strength: Jacina denoising-a (1-20)
        """
        gray = self.to_grayscale(img)
        denoised = cv2.fastNlMeansDenoising(gray, None, h=strength,
                                            templateWindowSize=7,
                                            searchWindowSize=21)
        logger.debug("Denoise primenjen")
        return denoised

    def binarize(self, img: np.ndarray, method: str = "adaptive") -> np.ndarray:
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
        else:  # otsu
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

        return binary

    def crop_roi(self, img: np.ndarray,
                 x: int, y: int, width: int, height: int,
                 padding: int = 5) -> np.ndarray:
        """
        Iseca Region of Interest (ROI) na osnovu koordinata iz template-a.

        Args:
            img: Ulazna slika
            x, y: Gornji levi ugao
            width, height: Dimenzije regiona
            padding: Dodatni padding oko regiona
        """
        h, w = img.shape[:2]

        # Dodaj padding ali ostani u granicama slike
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + width + padding)
        y2 = min(h, y + height + padding)

        roi = img[y1:y2, x1:x2]

        if roi.size == 0:
            raise ValueError(f"Prazan ROI: ({x1},{y1}) - ({x2},{y2})")

        logger.debug(f"Crop ROI: ({x1},{y1}) - ({x2},{y2}), velicina: {roi.shape}")
        return roi

    def remove_horizontal_lines(self, img: np.ndarray,
                                min_length: int = 50) -> np.ndarray:
        """
        Uklanja horizontalne linije (npr. linije za pisanje).
        Koristi morfoloske operacije.
        """
        gray = self.to_grayscale(img)
        binary = cv2.adaptiveThreshold(gray, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # Detektuj horizontalne linije
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (min_length, 1)
        )
        detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                                          horizontal_kernel, iterations=2)

        # Ukloni linije iz originalne slike
        repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        result = cv2.subtract(binary, detected_lines)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE,
                                  repair_kernel, iterations=1)

        # Vrati u normalan format (beli tekst na crnoj pozadini -> obrnuto)
        result = cv2.bitwise_not(result)

        logger.debug("Uklonjene horizontalne linije")
        return result

    def remove_vertical_lines(self, img: np.ndarray,
                              min_length: int = 30) -> np.ndarray:
        """
        Uklanja vertikalne linije (npr. linije kucica za JMBG).
        """
        gray = self.to_grayscale(img)
        binary = cv2.adaptiveThreshold(gray, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # Detektuj vertikalne linije
        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, min_length)
        )
        detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                                          vertical_kernel, iterations=2)

        # Ukloni linije
        result = cv2.subtract(binary, detected_lines)

        # Cleanup
        cleanup_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE,
                                  cleanup_kernel, iterations=1)

        result = cv2.bitwise_not(result)

        logger.debug("Uklonjene vertikalne linije")
        return result

    def remove_comb_boxes(self, img: np.ndarray,
                          cell_count: int = 13,
                          cell_width: int = 26) -> np.ndarray:
        """
        Uklanja vertikalne linije kucica (comb-boxes) kod JMBG polja.
        Ostavlja samo rukopis/cifre.

        Args:
            img: Slika JMBG polja
            cell_count: Broj kucica (13 za JMBG)
            cell_width: Sirina jedne kucice u pikselima
        """
        gray = self.to_grayscale(img)
        h, w = gray.shape[:2]

        binary = cv2.adaptiveThreshold(gray, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # Detektuj vertikalne linije (visina = 1/5 slike - manje tolerantno)
        # Smanjujemo visinu kernela da ne bi hvatali delove cifara (npr 1, 7)
        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, h // 5)
        )
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                                          vertical_kernel, iterations=2)

        # Detektuj horizontalne linije
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (w // 2, 1)
        )
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                                            horizontal_kernel, iterations=2)

        # Kombinuj linije
        all_lines = cv2.add(vertical_lines, horizontal_lines)
        
        # Smanjujemo dilation agression - koristimo krstasti kernel (cross) umesto punog pravougaonika
        # Ovo bolje cuva uglove cifara
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        all_lines = cv2.dilate(all_lines, dilate_kernel, iterations=1)

        # Ukloni linije iz binarizovane slike
        result = cv2.subtract(binary, all_lines)

        # Cleanup - povezi slomljene karaktere (horizontalno)
        # Koristimo 3x1 kernel da povezemo prekinute cifre a da ne spojimo vertikalno
        cleanup_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE,
                                  cleanup_kernel, iterations=1)
        
        # Jos jedan pass ciscenja suma
        result = cv2.fastNlMeansDenoising(result, None, 20, 7, 21)
        
        # Vrati u normalan format
        result = cv2.bitwise_not(result)

        logger.info(f"Uklonjene kucice ({cell_count} celija) - Aggressive mode")
        return result

    def enhance_contrast(self, img: np.ndarray,
                         clip_limit: float = 2.0) -> np.ndarray:
        """
        Pojacava kontrast koristeci CLAHE (Contrast Limited Adaptive Histogram Equalization).
        """
        gray = self.to_grayscale(img)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        logger.debug("Kontrast pojacan (CLAHE)")
        return enhanced

    def perspective_transform(self, img: np.ndarray) -> np.ndarray:
        """
        Ispravlja perspektivu dokumenta.
        Detektuje ivice papira i transformise na standardnu A4 proporciju.
        """
        gray = self.to_grayscale(img)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 75, 200)

        # Pronadji konture
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Sortiraj po povrsini, uzmi najvecu
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        document_contour = None
        for contour in contours[:5]:  # Proveri top 5
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) == 4:  # Cetvorougao
                document_contour = approx
                break

        if document_contour is None:
            logger.warning("Perspective transform: nije pronadjen dokument cetvorougao")
            return img

        # Proveri velicinu konture u odnosu na sliku
        img_area = img.shape[0] * img.shape[1]
        contour_area = cv2.contourArea(document_contour)
        min_area_ratio = 0.5  # Dokument mora zauzimati bar 50% slike

        if contour_area < img_area * min_area_ratio:
            logger.warning(f"Perspective transform: pronadjen dokument je premali ({contour_area/img_area:.2%}), preskacemo. Verovatno je detektovan samo deo obrasca.")
            return img

        # Sortiraj tacke: top-left, top-right, bottom-right, bottom-left
        pts = document_contour.reshape(4, 2)
        rect = self._order_points(pts)

        # Odredi dimenzije novog dokumenta
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # Destinacione tacke
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        # Transformacija
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

        logger.info("Perspective transform primenjen")
        return warped

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Sortira 4 tacke u redosledu: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype="float32")

        # Top-left ima najmanju sumu, bottom-right ima najvecu
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # Top-right ima najmanju razliku, bottom-left ima najvecu
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def detect_anchor(self, img: np.ndarray,
                      anchor_template: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Detektuje anchor element (sidro) na slici koristeci template matching.

        Args:
            img: Ulazna slika
            anchor_template: Template slike sidra (npr. logo "РЗС")

        Returns:
            (x, y) koordinate centra sidra ili None
        """
        gray = self.to_grayscale(img)
        template_gray = self.to_grayscale(anchor_template)

        result = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val > 0.7:  # Threshold za pouzdanu detekciju
            th, tw = template_gray.shape[:2]
            center_x = max_loc[0] + tw // 2
            center_y = max_loc[1] + th // 2
            logger.info(f"Anchor detektovan na ({center_x}, {center_y}), confidence: {max_val:.2f}")
            return (center_x, center_y)

        logger.warning(f"Anchor nije detektovan (max confidence: {max_val:.2f})")
        return None

    def resize_to_dpi(self, img: np.ndarray,
                      current_dpi: int = 150) -> np.ndarray:
        """
        Skalira sliku na target DPI.
        """
        if current_dpi == self.target_dpi:
            return img

        scale = self.target_dpi / current_dpi
        new_width = int(img.shape[1] * scale)
        new_height = int(img.shape[0] * scale)

        resized = cv2.resize(img, (new_width, new_height),
                            interpolation=cv2.INTER_CUBIC)

        logger.info(f"Slika skalirana sa {current_dpi} DPI na {self.target_dpi} DPI")
        return resized

    def resize_to_width(self, img: np.ndarray, target_width: int) -> np.ndarray:
        """
        Skalira sliku na zadatu sirinu uz ocuvanje aspect ratio-a.
        """
        h, w = img.shape[:2]
        if w == target_width:
            return img
        
        scale = target_width / w
        new_height = int(h * scale)
        
        resized = cv2.resize(img, (target_width, new_height), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        logger.info(f"Resize: {w}x{h} -> {target_width}x{new_height} (scale={scale:.2f})")
        return resized

    def sharpen(self, img: np.ndarray) -> np.ndarray:
        """
        Izostruje sliku koristeci unsharp masking.
        """
        gray = self.to_grayscale(img)
        blurred = cv2.GaussianBlur(gray, (0, 0), 3)
        sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        logger.debug("Sharpen primenjen")
        return sharpened

    def process_field(self, img: np.ndarray,
                      field_config: dict) -> np.ndarray:
        """
        Primenjuje odgovarajuce preprocessing korake za dato polje.

        Args:
            img: Cela slika dokumenta
            field_config: Konfiguracija polja iz template-a

        Returns:
            Obradjena ROI slika polja
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
            elif step == 'remove_vertical_lines':
                roi = self.remove_vertical_lines(roi)
            elif step == 'remove_comb_boxes':
                comb_config = field_config.get('comb_box', {})
                roi = self.remove_comb_boxes(
                    roi,
                    cell_count=comb_config.get('cell_count', 13),
                    cell_width=comb_config.get('cell_width', 26)
                )
            elif step == 'enhance_contrast':
                roi = self.enhance_contrast(roi)
            elif step == 'sharpen':
                roi = self.sharpen(roi)

        # Finalna binarizacija za OCR
        roi = self.binarize(roi)

        return roi

    def preprocess_full_document(self, img: np.ndarray) -> np.ndarray:
        """
        Kompletna priprema dokumenta pre obrade polja.

        Args:
            img: Originalna slika dokumenta

        Returns:
            Pripremljena slika
        """
        # 1. Perspective transform (ako je potrebno)
        processed = self.perspective_transform(img)

        # 2. Deskew
        processed = self.deskew(processed)

        # 3. Denoise (blago)
        processed = self.denoise(processed, strength=5)

        logger.info("Dokument pripremljen za OCR")
        return processed


class PDFProcessor:
    """
    Processor za PDF dokumente.
    Konvertuje PDF stranice u slike za OCR.
    """

    def __init__(self, dpi: int = 300):
        self.dpi = dpi

    def pdf_to_images(self, pdf_path: str) -> List[np.ndarray]:
        """
        Konvertuje PDF u listu slika.

        Args:
            pdf_path: Putanja do PDF fajla

        Returns:
            Lista numpy array-eva (slika)
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF nije instaliran. Instaliraj sa: pip install PyMuPDF")

        images = []
        doc = fitz.open(pdf_path)

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)

            # Konvertuj u sliku sa zeljenim DPI-em
            zoom = self.dpi / 72  # 72 je default DPI za PDF
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)

            # Konvertuj u numpy array
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )

            # Konvertuj u BGR ako je RGBA
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            images.append(img)
            logger.info(f"PDF stranica {page_num + 1} konvertovana, dimenzije: {img.shape}")

        doc.close()
        return images

    def pdf_to_images_from_bytes(self, pdf_bytes: bytes) -> List[np.ndarray]:
        """
        Konvertuje PDF iz bajtova u listu slika.
        """
        try:
            import fitz
        except ImportError:
            raise ImportError("PyMuPDF nije instaliran")

        images = []
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            zoom = self.dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)

            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )

            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            images.append(img)

        doc.close()
        return images
