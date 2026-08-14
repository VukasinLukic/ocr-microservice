# SV-20 OCR Mikroservis

Offline Python OCR mikroservis za prepoznavanje podataka sa SV-20 obrazaca studenata.

## Pregled

Ovaj servis je deo sistema za automatizovanu obradu SV-20 obrazaca na Fakultetu organizacionih nauka. Koristi EasyOCR sa podrskom za srpsku cirilicu i latinicu, OpenCV za image processing, i AMD DirectML za GPU akceleraciju.

## Arhitektura

```
┌─────────────────────────────────────────────────────────────┐
│                      JAVA SISTEM                            │
│  ┌─────────────┐         Socket          ┌─────────────┐   │
│  │   KLIJENT   │ ◄──────────────────────►│   SERVER    │   │
│  │  (Swing UI) │       Port 9000         │  (Kontroler)│   │
│  └─────────────┘                         └──────┬──────┘   │
│                                                 │          │
│                                          ┌──────▼──────┐   │
│                                          │    MySQL    │   │
│                                          │   bazaocr   │   │
│                                          └─────────────┘   │
└─────────────────────────────────────────────────────────────┘
                          │
                          │ HTTP REST API (Port 9001)
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                 PYTHON OCR MIKROSERVIS                      │
│                    (Ovaj projekat)                          │
│  FastAPI + EasyOCR + OpenCV + AMD DirectML                  │
└─────────────────────────────────────────────────────────────┘
```

## Funkcionalnosti

- **OCR** - Prepoznavanje teksta (srpska cirilica i latinica)
- **OMR** - Detekcija zaokruzenih opcija (pol, status, finansiranje)
- **Image Processing** - Deskew, denoise, binarization, ROI cropping
- **Validacija** - JMBG (mod 11), datumi, broj indeksa
- **GPU Akceleracija** - AMD DirectML podrska
- **Offline rad** - Svi modeli lokalno

## Struktura Projekta

```
sv20-ocr-service/
├── main.py                      # Entry point
├── config.py                    # Konfiguracija
├── requirements.txt             # Dependencies
├── download_models.py           # Offline priprema modela
├── README.md                    # Dokumentacija
│
├── api/
│   ├── __init__.py
│   └── server.py                # FastAPI endpoints
│
├── processors/
│   ├── __init__.py
│   ├── image_processor.py       # OpenCV pipeline
│   ├── ocr_engine.py            # EasyOCR wrapper
│   ├── omr_logic.py             # OMR za checkboxove
│   └── validators.py            # Validacija formata
│
├── templates/
│   └── sv20_template.json       # Koordinate polja
│
├── models/                      # EasyOCR modeli (offline)
│
├── logs/                        # Log fajlovi
│
└── tests/
    └── test_*.py                # pytest testovi (validators, OMR logika)
```

> **VAŽNO:** `templates/sv20_template.json` je **jedini i zvaničan izvor** definicije
> polja (koordinate, tip, validacija, OMR opcije) za ovaj servis - ne MySQL baza.
> `config.py` sadrži `DB_CONFIG` kao ostatak ranijeg plana da se polja čitaju iz
> `tippolja` tabele, ali ta konekcija nigde nije povezana/korišćena. Ako Java/DB
> strana promeni ili doda polje u `tippolja`, OCR servis to **neće videti** dok se
> ista izmena ručno ne napravi i u `sv20_template.json` (preporučeno preko
> Template Editora na `/editor`, ne ručnim uređivanjem JSON-a). Dva izvora istine
> za isti podatak - ovo je jedini koji ovaj servis stvarno čita.

## Instalacija

### 1. Kreiranje virtualnog okruzenja

```bash
cd sv20-ocr-service
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 2. Instalacija dependencies

```bash
pip install -r requirements.txt
```

### 3. Preuzimanje modela (za offline rad)

```bash
python download_models.py
```

## Pokretanje

```bash
python main.py
```

Servis ce biti dostupan na:
- **API**: http://127.0.0.1:9001
- **Dokumentacija**: http://127.0.0.1:9001/docs
- **Health Check**: http://127.0.0.1:9001/api/health

### Opcije pokretanja

```bash
python main.py --help

# Sa custom portom
python main.py --port 8080

# Sa auto-reload (development)
python main.py --reload

# Sa debug logovima
python main.py --log-level debug
```

## API Endpoints

### Health Check
```
GET /api/health
```
Vraca status servisa i informacije o GPU/OCR.

### Template Info
```
GET /api/template
```
Vraca definiciju polja za SV-20 obrazac.

### OCR Processing
```
POST /api/ocr/process
Content-Type: multipart/form-data

file: <slika obrasca>
obrazac_id: <opcioni ID>
page_number: <broj stranice za PDF>
```

### Response Format
```json
{
  "success": true,
  "message": "Obrada zavrsena",
  "obrazac_id": "123",
  "total_fields": 12,
  "successful_fields": 10,
  "failed_fields": 2,
  "processing_time_ms": 1500.5,
  "fields": [
    {
      "field_id": 1,
      "field_name": "ime_prezime_studenta",
      "field_type": "TEXT",
      "ocr_value": "Вукашин Лукић",
      "validated_value": "Вукашин Лукић",
      "confidence": 94.5,
      "is_valid": true,
      "validation_error": null,
      "coordinates": {
        "x": 80,
        "y": 180,
        "width": 350,
        "height": 50
      }
    }
  ]
}
```

## Polja SV-20 Obrasca

| ID | Naziv | Tip | Opis |
|----|-------|-----|------|
| 1 | ime_prezime_studenta | TEXT | Ime i prezime |
| 2 | jmbg | NUMERIC | JMBG (13 cifara) |
| 3 | broj_indeksa | ALPHANUMERIC | Format: GGGG/BBBB |
| 16 | pol | OMR_SINGLE | Muski (1) / Zenski (2) |
| ... | ... | ... | ... |

## Validacije

### JMBG
- 13 cifara
- Kontrolna cifra po modulu 11
- Validacija dana i meseca

### Broj indeksa
- Format: GGGG/BBBB (npr. 2023/0342)
- Automatska korekcija O->0, l->1

### Datum
- Format: DD.MM.YYYY
- Validacija opsega godine

## GPU Podrska

Servis automatski detektuje dostupan GPU:

1. **AMD DirectML** - onnxruntime-directml
2. **NVIDIA CUDA** - onnxruntime-gpu
3. **CPU Fallback** - ako GPU nije dostupan

## Integracija sa Java Klijentom

```java
// Primer poziva iz Java klijenta
OCRService ocrService = new OCRService();

if (ocrService.isServiceAvailable()) {
    OCRResult result = ocrService.processImage(
        "putanja/do/slike.jpg",
        obrazacId
    );

    for (OCRFieldResult field : result.getFields()) {
        // Kreiraj StavkeObrasca objekat
        StavkeObrasca stavka = new StavkeObrasca();
        stavka.setOcrVrednost(field.getOcrValue());
        stavka.setNivoPodudarnosti(field.getConfidence());
        stavka.setOcrUspesno(field.isValid());
        // ...
    }
}
```

## Troubleshooting

### OCR ne prepoznaje tekst
- Proveri kvalitet slike (min 300 DPI)
- Proveri da slika nije previše tamna/svetla
- Pokusaj sa manjim ROI regionom

### GPU nije detektovan
- Instaliraj AMD Adrenalin drajvere
- Proveri da je onnxruntime-directml instaliran
- Restartuj Python interpreter

### Modeli se stalno preuzimaju
- Pokrenuti `download_models.py` jednom
- Proveriti `models/` direktorijum

## Development

### Pokretanje testova
```bash
pytest tests/
```

### Formatiranje koda
```bash
black .
flake8 .
```

## Autor

**Vukasin Lukic**
- Projekat: Seminarski rad - Softversko inzenjerstvo
- Fakultet: Fakultet organizacionih nauka, Beograd
- Skolska godina: 2024/2025

## Licenca

Ovaj projekat je deo seminarskog rada i namenjen je iskljucivo edukativnim svrhama.
