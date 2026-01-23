"""
SV-20 OCR Mikroservis - FastAPI Server
REST API endpoints za OCR obradu SV-20 obrazaca.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import json
import logging
import tempfile
import os
from pathlib import Path
import time

# Import procesora
import sys
import cv2
import base64
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))

from processors.image_processor import ImageProcessor, PDFProcessor
from processors.ocr_engine import OCREngineFactory
from processors.omr_logic import OMRProcessor
from processors.validators import FieldValidator, postprocess_text
from config import Config

# Konfiguracija loggera
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Kreiranje FastAPI aplikacije
app = FastAPI(
    title="SV-20 OCR Mikroservis",
    description="Offline OCR servis za prepoznavanje podataka sa SV-20 obrazaca studenata",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware - dozvoli pozive sa Java klijenta
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for Template Creator
app.mount("/editor", StaticFiles(directory="static", html=True), name="static")


# Pydantic modeli za response
class FieldCoordinates(BaseModel):
    x: int
    y: int
    width: int
    height: int


class OCRFieldResult(BaseModel):
    field_id: int
    field_name: str
    field_type: str
    ocr_value: str
    validated_value: str
    confidence: float
    is_valid: bool
    validation_error: Optional[str] = None
    coordinates: Optional[FieldCoordinates] = None
    omr_detected: Optional[bool] = None


class OCRResponse(BaseModel):
    success: bool
    message: str
    obrazac_id: Optional[str] = None
    total_fields: int
    successful_fields: int
    failed_fields: int
    processing_time_ms: float
    fields: List[OCRFieldResult]


class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    gpu_backend: str
    ocr_initialized: bool
    languages: List[str]


class TemplateResponse(BaseModel):
    document: Dict[str, Any]
    fields: List[Dict[str, Any]]


# Globalne instance procesora
image_processor: Optional[ImageProcessor] = None
omr_processor: Optional[OMRProcessor] = None
pdf_processor: Optional[PDFProcessor] = None
template_data: Optional[Dict] = None


def load_template() -> dict:
    """Ucitava template za SV-20 obrazac."""
    global template_data
    if template_data is None:
        template_path = Config.TEMPLATES_DIR / "sv20_template.json"
        if not template_path.exists():
            raise FileNotFoundError(f"Template fajl ne postoji: {template_path}")
        with open(template_path, 'r', encoding='utf-8') as f:
            template_data = json.load(f)
    return template_data


@app.on_event("startup")
async def startup_event():
    """Inicijalizacija pri pokretanju servera."""
    global image_processor, omr_processor, pdf_processor

    logger.info("=" * 50)
    logger.info("Pokrecem SV-20 OCR Mikroservis...")
    logger.info("=" * 50)

    # Osiguraj da direktorijumi postoje
    Config.ensure_directories()

    # Inicijalizuj procesore
    image_processor = ImageProcessor(target_dpi=Config.TARGET_DPI)
    omr_processor = OMRProcessor()
    pdf_processor = PDFProcessor(dpi=Config.TARGET_DPI)

    logger.info("Image procesori inicijalizovani")

    # Inicijalizuj OCR Engine
    logger.info("Inicijalizujem OCR Engine...")
    try:
        OCREngineFactory.get_instance(
            languages=Config.OCR_LANGUAGES,
            model_storage_directory=str(Config.MODELS_DIR),
            use_gpu=Config.USE_GPU,
            gpu_backend=Config.GPU_BACKEND
        )
        ocr_info = OCREngineFactory.get_instance().get_info()
        logger.info(f"OCR Engine spreman: {ocr_info}")
    except Exception as e:
        logger.error(f"Greska pri inicijalizaciji OCR Engine-a: {e}")
        logger.warning("Servis ce raditi bez OCR - samo za testiranje")

    # Ucitaj template
    try:
        load_template()
        logger.info("Template ucitan uspesno")
    except Exception as e:
        logger.error(f"Greska pri ucitavanju template-a: {e}")

    logger.info("=" * 50)
    logger.info(f"Servis spreman na http://{Config.HOST}:{Config.PORT}")
    logger.info("=" * 50)


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Vraca status servisa i informacije o GPU/OCR.
    """
    gpu_backend = Config.detect_gpu()
    ocr_initialized = OCREngineFactory.is_initialized()

    languages = []
    if ocr_initialized:
        languages = OCREngineFactory.get_instance().languages

    return HealthResponse(
        status="healthy",
        gpu_available=gpu_backend != "cpu",
        gpu_backend=gpu_backend,
        ocr_initialized=ocr_initialized,
        languages=languages
    )


@app.get("/api/template", response_model=TemplateResponse)
async def get_template():
    """
    Vraca trenutni template za SV-20 obrazac.
    Korisno za Java klijent da zna koja polja postoje.
    """
    try:
        template = load_template()
        return TemplateResponse(
            document=template.get('document', {}),
            fields=template.get('fields', [])
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/template/save")
async def save_template(template: Dict[str, Any]):
    """
    Cuva azurirani template na disk.
    """
    try:
        template_path = Config.TEMPLATES_DIR / "sv20_template.json"
        
        # Validacija strukture (basic)
        if "fields" not in template or "document" not in template:
            raise HTTPException(status_code=400, detail="Invalid template structure")
            
        with open(template_path, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2, ensure_ascii=False)
            
        # Reload global template cache
        global template_data
        template_data = template
        
        logger.info("Template uspesno sacuvan i reloadovan.")
        return {"success": True, "message": "Template saved successfully"}
        
    except Exception as e:
        logger.error(f"Greska pri cuvanju template-a: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/fields")
async def get_fields():
    """
    Vraca listu polja iz template-a (pojednostavljena verzija).
    """
    try:
        template = load_template()
        fields = []
        for field in template.get('fields', []):
            fields.append({
                'id': field['id'],
                'name': field['name'],
                'label': field.get('label', field['name']),
                'type': field['type'],
                'required': field.get('required', False),
                'page': field.get('page', 1)
            })
        return {"fields": fields, "count": len(fields)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def image_to_base64(img) -> str:
    """Konvertuje OpenCV sliku u base64 string."""
    import cv2
    import base64
    
    success, buffer = cv2.imencode('.jpg', img)
    if not success:
        raise ValueError("Could not encode image")
    
    img_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_str}"


# Global state for the editor (single user)
LAST_UPLOADED_PDF_PATH = None

@app.post("/api/template/render-upload")
async def render_upload_for_editor(file: UploadFile = File(...)):
    """
    Prihvata sliku ili PDF, renderuje prvu stranicu i vraca je kao base64 string
    za prikaz u editoru.
    """
    global LAST_UPLOADED_PDF_PATH
    
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.pdf', '.tif', '.tiff', '.bmp'}
    file_ext = Path(file.filename).suffix.lower() if file.filename else ''
    
    if file_ext not in allowed_extensions:
        raise HTTPException(400, f"Unsupported file type: {file_ext}")

    tmp_path = None
    try:
        # Sacuvaj uploadovani fajl privremeno
        content = await file.read()
        # Cuvamo u persistent temp fajlu da bismo mogli kasnije da citamo druge strane
        # Koristimo standardni temp dir ali ne brisemo odmah
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        LAST_UPLOADED_PDF_PATH = tmp_path

        # Detect PDF vs Image
        if file_ext == '.pdf':
            # Koristi postojeci pdf_processor
            if not pdf_processor:
                 raise HTTPException(503, "PDF processor not initialized")
            
            # Uzmi PRVU stranu po defaultu
            images = pdf_processor.pdf_to_images(tmp_path)
            if not images:
                raise HTTPException(400, "Empty PDF")
            img = images[0]
        else:
            # Obicna slika
            img = image_processor.load_image(tmp_path)
            
        # Resize to 1024px width (Editor standard)
        img = image_processor.resize_to_width(img, 1024)
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', img)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "image_data": f"data:image/jpeg;base64,{img_str}",
            "is_pdf": file_ext == '.pdf',
            "filename": file.filename
        }

    except Exception as e:
        logger.error(f"Error rendering upload: {e}")
        if tmp_path and os.path.exists(tmp_path) and LAST_UPLOADED_PDF_PATH != tmp_path:
             os.unlink(tmp_path)
        raise HTTPException(500, str(e))

@app.get("/api/template/render-page")
async def render_page_for_editor(page: int = 1):
    """
    Renderuje specificnu stranicu poslednjeg uploadovanog PDF-a.
    """
    global LAST_UPLOADED_PDF_PATH
    
    logger.info(f"Render Page Requested: {page}")
    logger.info(f"Last PDF Path: {LAST_UPLOADED_PDF_PATH}")

    if not LAST_UPLOADED_PDF_PATH or not os.path.exists(LAST_UPLOADED_PDF_PATH):
        logger.error("No PDF uploaded or file missing")
        raise HTTPException(404, "No PDF uploaded yet")
        
    try:
        # Proveri ekstenziju
        if not LAST_UPLOADED_PDF_PATH.lower().endswith('.pdf'):
             raise HTTPException(400, "Last uploaded file is not a PDF")

        if not pdf_processor:
             logger.error("PDF processor not initialized")
             raise HTTPException(503, "PDF processor not initialized")
             
        # Ucitaj slike
        logger.info(f"Converting PDF to images: {LAST_UPLOADED_PDF_PATH}")
        images = pdf_processor.pdf_to_images(LAST_UPLOADED_PDF_PATH)
        
        if not images:
            logger.error("No images extracted from PDF")
            raise HTTPException(500, "Failed to extract images from PDF")

        logger.info(f"Extracted {len(images)} images")
        
        if page < 1 or page > len(images):
            raise HTTPException(400, f"Page {page} out of range (1-{len(images)})")
            
        # Uzmi trazenu stranu (indeks je page-1)
        img = images[page - 1]
        
        # Resize to 1024px width (Editor standard)
        logger.info("Resizing image...")
        img = image_processor.resize_to_width(img, 1024)
        
        # Convert to base64
        logger.info("Encoding to base64...")
        _, buffer = cv2.imencode('.jpg', img)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        logger.info("Returning response")
        return {
            "image_data": f"data:image/jpeg;base64,{img_str}",
            "page": page
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rendering page {page}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(500, f"Internal Server Error: {str(e)}")


@app.post("/api/ocr/process", response_model=OCRResponse)
async def process_image(
    file: UploadFile = File(...),
    obrazac_id: Optional[str] = Form(None),
    page_number: Optional[int] = Form(None)
):
    """
    Procesira sliku SV-20 obrasca i vraca ekstrahovane podatke.

    - **file**: Slika obrasca (JPG, PNG, PDF)
    - **obrazac_id**: Opcioni ID obrasca za pracenje
    - **page_number**: Broj stranice za obradu (None = sve strane za PDF)

    Vraca JSON sa ekstrahovanim poljima, confidence skorovima i validacijom.
    """
    start_time = time.time()
    logger.info(f"Primljen zahtev za OCR: {file.filename}, obrazac_id={obrazac_id}")

    # Validacija tipa fajla
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.pdf', '.tif', '.tiff', '.bmp'}
    file_ext = Path(file.filename).suffix.lower() if file.filename else ''
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Nepodrzani format fajla '{file_ext}'. Dozvoljeni: {', '.join(allowed_extensions)}"
        )

    tmp_path = None
    try:
        # Sacuvaj uploadovani fajl privremeno
        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        # Ucitaj template
        template = load_template()

        # High-Res OCR Implementation
        TARGET_WIDTH = 2048
        TEMPLATE_WIDTH = 1024
        scale_factor = TARGET_WIDTH / TEMPLATE_WIDTH

        # Dobavi OCR engine
        if not OCREngineFactory.is_initialized():
            raise HTTPException(
                status_code=503,
                detail="OCR Engine nije inicijalizovan"
            )
        ocr_engine = OCREngineFactory.get_instance()

        results = []
        successful = 0
        failed = 0

        if file_ext == '.pdf':
            # PDF - konvertuj SVE strane u slike
            all_images = pdf_processor.pdf_to_images(tmp_path)
            total_pages = len(all_images)
            logger.info(f"PDF ima {total_pages} stranica")

            # Odredite koje strane obraditi
            if page_number is not None:
                # Samo jedna specificna strana
                if page_number > total_pages:
                    raise HTTPException(
                        status_code=400,
                        detail=f"PDF ima {total_pages} stranica, trazena je {page_number}"
                    )
                pages_to_process = [page_number]
            else:
                # Sve strane
                pages_to_process = list(range(1, total_pages + 1))

            # Obradi svaku stranu
            for current_page in pages_to_process:
                logger.info(f"Obrada stranice {current_page}/{total_pages}")
                img = all_images[current_page - 1]
                img = image_processor.resize_to_width(img, TARGET_WIDTH)
                img = image_processor.preprocess_full_document(img)

                # Filtriraj polja za trenutnu stranicu
                page_fields = [f for f in template['fields']
                              if f.get('page', 1) == current_page]

                for field in page_fields:
                    try:
                        # Scale coordinates for High-Res Image
                        scaled_field = field.copy()
                        if 'coordinates' in scaled_field:
                            c = scaled_field['coordinates']
                            scaled_field['coordinates'] = {
                                'x': int(c['x'] * scale_factor),
                                'y': int(c['y'] * scale_factor),
                                'width': int(c['width'] * scale_factor),
                                'height': int(c['height'] * scale_factor)
                            }

                        field_result = process_single_field(img, scaled_field, ocr_engine)

                        # Restore original coordinates for the response
                        if 'coordinates' in field:
                            field_result.coordinates = FieldCoordinates(**field['coordinates'])

                        results.append(field_result)

                        if field_result.is_valid:
                            successful += 1
                        else:
                            failed += 1

                    except Exception as e:
                        logger.error(f"Greska pri obradi polja {field['name']} (strana {current_page}): {e}")
                        results.append(OCRFieldResult(
                            field_id=field['id'],
                            field_name=field['name'],
                            field_type=field.get('type', 'TEXT'),
                            ocr_value="",
                            validated_value="",
                            confidence=0.0,
                            is_valid=False,
                            validation_error=str(e),
                            coordinates=FieldCoordinates(**field['coordinates']) if 'coordinates' in field else None
                        ))
                        failed += 1

        else:
            # Obicna slika - obradi kao jednu stranicu
            img = image_processor.load_image(tmp_path)
            img = image_processor.resize_to_width(img, TARGET_WIDTH)
            img = image_processor.preprocess_full_document(img)

            # Za slike, obradi samo polja sa page=1 (ili bez page atributa)
            target_page = page_number if page_number else 1
            page_fields = [f for f in template['fields']
                          if f.get('page', 1) == target_page]

            for field in page_fields:
                try:
                    scaled_field = field.copy()
                    if 'coordinates' in scaled_field:
                        c = scaled_field['coordinates']
                        scaled_field['coordinates'] = {
                            'x': int(c['x'] * scale_factor),
                            'y': int(c['y'] * scale_factor),
                            'width': int(c['width'] * scale_factor),
                            'height': int(c['height'] * scale_factor)
                        }

                    field_result = process_single_field(img, scaled_field, ocr_engine)

                    if 'coordinates' in field:
                        field_result.coordinates = FieldCoordinates(**field['coordinates'])

                    results.append(field_result)

                    if field_result.is_valid:
                        successful += 1
                    else:
                        failed += 1

                except Exception as e:
                    logger.error(f"Greska pri obradi polja {field['name']}: {e}")
                    results.append(OCRFieldResult(
                        field_id=field['id'],
                        field_name=field['name'],
                        field_type=field.get('type', 'TEXT'),
                        ocr_value="",
                        validated_value="",
                        confidence=0.0,
                        is_valid=False,
                        validation_error=str(e),
                        coordinates=FieldCoordinates(**field['coordinates']) if 'coordinates' in field else None
                    ))
                    failed += 1

        processing_time = (time.time() - start_time) * 1000

        logger.info(f"OCR zavrseno: {successful} uspesno, {failed} neuspesno, "
                   f"vreme: {processing_time:.0f}ms")

        return OCRResponse(
            success=True,
            message="Obrada zavrsena",
            obrazac_id=obrazac_id,
            total_fields=len(results),
            successful_fields=successful,
            failed_fields=failed,
            processing_time_ms=processing_time,
            fields=results
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Greska pri OCR obradi: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Obrisi privremeni fajl
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def process_single_field(img, field_config: dict, ocr_engine) -> OCRFieldResult:
    """
    Procesira pojedinacno polje iz obrasca.

    Args:
        img: Pripremljena slika dokumenta
        field_config: Konfiguracija polja iz template-a
        ocr_engine: OCR engine instanca

    Returns:
        OCRFieldResult sa rezultatima
    """
    field_type = field_config.get('type', 'TEXT')
    field_name = field_config.get('name', '')
    coords = field_config.get('coordinates', {})

    # Specijalna obrada za OMR polja
    if field_type.startswith('OMR'):
        return process_omr_field(img, field_config)

    # Specijalna obrada za SIGNATURE polje
    if field_type == 'SIGNATURE':
        return OCRFieldResult(
            field_id=field_config['id'],
            field_name=field_name,
            field_type=field_type,
            ocr_value="[POTPIS]",
            validated_value="[POTPIS]",
            confidence=1.0,
            is_valid=True,
            validation_error=None,
            coordinates=FieldCoordinates(**coords) if coords else None
        )

    # Obrada OCR polja
    try:
        # Pripremi ROI
        processed_roi = image_processor.process_field(img, field_config)

        # OCR
        if field_type == 'NUMERIC' and field_config.get('validation', {}).get('length') == 13:
            # JMBG - samo cifre
            ocr_value, confidence = ocr_engine.recognize_digits_only(processed_roi)
        elif field_type == 'NUMERIC':
            # Ostala numericka polja
            ocr_value, confidence = ocr_engine.recognize_digits_only(processed_roi)
        else:
            # Tekstualna polja
            ocr_value, confidence = ocr_engine.recognize_single_field(processed_roi)

        # Postprocessing
        postprocess_ops = field_config.get('postprocessing', [])
        if postprocess_ops:
            ocr_value = postprocess_text(ocr_value, postprocess_ops)

        # Validacija
        validation_result = FieldValidator.validate_field(ocr_value, field_config)

        return OCRFieldResult(
            field_id=field_config['id'],
            field_name=field_name,
            field_type=field_type,
            ocr_value=ocr_value,
            validated_value=validation_result['cleaned_value'],
            confidence=confidence * 100,  # Konvertuj u procenat
            is_valid=validation_result['is_valid'],
            validation_error=validation_result['error'],
            coordinates=FieldCoordinates(**coords) if coords else None
        )

    except Exception as e:
        logger.error(f"Greska u process_single_field za {field_name}: {e}")
        raise


def process_omr_field(img, field_config: dict) -> OCRFieldResult:
    """
    Procesira OMR polje (zaokruzene opcije, checkboxovi).

    Args:
        img: Slika dokumenta
        field_config: Konfiguracija polja

    Returns:
        OCRFieldResult
    """
    field_type = field_config.get('type', 'OMR_SINGLE')
    field_name = field_config.get('name', '')
    coords = field_config.get('coordinates', {})
    omr_options = field_config.get('omr_options', [])

    try:
        # Iseci ROI
        roi = image_processor.crop_roi(
            img,
            coords['x'], coords['y'],
            coords['width'], coords['height']
        )

        if field_type == 'OMR_MULTI':
            # Visestruki izbor
            result = omr_processor.detect_multi_select(roi, omr_options)
            ocr_value = ', '.join(result.get('values', []))
            validated_value = ocr_value
        else:
            # Jednostruki izbor
            result = omr_processor.detect_marked_option(roi, omr_options)
            ocr_value = result.get('value', '') or ''
            validated_value = ocr_value

        confidence = result.get('confidence', 0.0) * 100

        return OCRFieldResult(
            field_id=field_config['id'],
            field_name=field_name,
            field_type=field_type,
            ocr_value=ocr_value,
            validated_value=validated_value,
            confidence=confidence,
            is_valid=bool(ocr_value),
            validation_error=result.get('warning'),
            coordinates=FieldCoordinates(**coords) if coords else None,
            omr_detected=True
        )

    except Exception as e:
        logger.error(f"Greska u process_omr_field za {field_name}: {e}")
        return OCRFieldResult(
            field_id=field_config['id'],
            field_name=field_name,
            field_type=field_type,
            ocr_value="",
            validated_value="",
            confidence=0.0,
            is_valid=False,
            validation_error=str(e),
            coordinates=FieldCoordinates(**coords) if coords else None,
            omr_detected=False
        )


@app.post("/api/ocr/process-raw")
async def process_image_raw(
    file: UploadFile = File(...),
    obrazac_id: Optional[str] = Form(None)
):
    """
    Procesira sliku bez template-a - vraca sve prepoznate tekst blokove.
    Korisno za debug i kreiranje template-a.
    """
    start_time = time.time()

    try:
        content = await file.read()

        # Ucitaj sliku
        img = image_processor.load_image_from_bytes(content)
        
        # Standardizuj velicinu (1024px)
        img = image_processor.resize_to_width(img, 1024)
        
        img = image_processor.preprocess_full_document(img)

        # OCR celog dokumenta
        if not OCREngineFactory.is_initialized():
            raise HTTPException(status_code=503, detail="OCR Engine nije inicijalizovan")

        ocr_engine = OCREngineFactory.get_instance()
        results = ocr_engine.recognize(img, detail=1)

        processing_time = (time.time() - start_time) * 1000

        return {
            "success": True,
            "obrazac_id": obrazac_id,
            "processing_time_ms": processing_time,
            "text_blocks": [
                {
                    "text": r['text'],
                    "confidence": r['confidence'],
                    "bbox": r['bbox']
                }
                for r in results
            ],
            "total_blocks": len(results)
        }

    except Exception as e:
        logger.error(f"Greska pri raw OCR: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/info")
async def get_info():
    """
    Vraca informacije o servisu.
    """
    ocr_info = {}
    if OCREngineFactory.is_initialized():
        ocr_info = OCREngineFactory.get_instance().get_info()

    return {
        "service": "SV-20 OCR Mikroservis",
        "version": "1.0.0",
        "host": Config.HOST,
        "port": Config.PORT,
        "ocr": ocr_info,
        "gpu_backend": Config.detect_gpu(),
        "target_dpi": Config.TARGET_DPI
    }


# Za direktno pokretanje
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=Config.HOST, port=Config.PORT)
