"""
SV-20 OCR Mikroservis - FastAPI Server
REST API endpoints za OCR obradu SV-20 obrazaca.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
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


@app.post("/api/ocr/process", response_model=OCRResponse)
async def process_image(
    file: UploadFile = File(...),
    obrazac_id: Optional[str] = Form(None),
    page_number: Optional[int] = Form(1)
):
    """
    Procesira sliku SV-20 obrasca i vraca ekstrahovane podatke.

    - **file**: Slika obrasca (JPG, PNG, PDF)
    - **obrazac_id**: Opcioni ID obrasca za pracenje
    - **page_number**: Broj stranice za obradu (default: 1)

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

        # Ucitaj sliku
        if file_ext == '.pdf':
            # PDF - konvertuj u slike
            images = pdf_processor.pdf_to_images(tmp_path)
            if page_number > len(images):
                raise HTTPException(
                    status_code=400,
                    detail=f"PDF ima {len(images)} stranica, trazena je {page_number}"
                )
            img = images[page_number - 1]
        else:
            # Obicna slika
            img = image_processor.load_image(tmp_path)

        # Pripremi dokument
        img = image_processor.preprocess_full_document(img)

        # Dobavi OCR engine
        if not OCREngineFactory.is_initialized():
            raise HTTPException(
                status_code=503,
                detail="OCR Engine nije inicijalizovan"
            )
        ocr_engine = OCREngineFactory.get_instance()

        # Procesiraj svako polje
        results = []
        successful = 0
        failed = 0

        # Filtriraj polja za trenutnu stranicu
        page_fields = [f for f in template['fields']
                      if f.get('page', 1) == page_number]

        for field in page_fields:
            try:
                field_result = process_single_field(img, field, ocr_engine)
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
            total_fields=len(page_fields),
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
