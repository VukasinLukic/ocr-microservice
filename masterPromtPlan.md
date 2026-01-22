# Instrukcije za Claude: Implementacija Python OCR Mikroservisa za SV-20 Obrazac

## 1. Kontekst Projekta

Radiš na implementaciji **Python OCR mikroservisa** koji treba da se integriše sa postojećom Java klijent-server aplikacijom za obradu SV-20 obrazaca (studentski formulari na Fakultetu organizacionih nauka).

### 1.1 Postojeća Arhitektura

```
┌─────────────────────────────────────────────────────────────┐
│                      JAVA SISTEM                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
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
│                    (Ti implementiraš)                       │
├─────────────────────────────────────────────────────────────┤
│  FastAPI + EasyOCR + OpenCV + AMD DirectML                  │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Moduli Java Projekta

- **KLIJENT**: Swing GUI, forme za unos podataka
- **SERVER**: Poslovna logika, komunikacija sa bazom
- **ZAJEDNICKI**: Domenski objekti, deljeni između klijenta i servera

---

## 2. Struktura Baze Podataka

### 2.1 Tabela: `sv20obrazac`

```sql
CREATE TABLE sv20obrazac (
    idObrazac INT PRIMARY KEY AUTO_INCREMENT,
    datumUnosa DATE,
    skolskaGodina INT,
    semestar INT,
    status ENUM('PODNET', 'U_OBRADI', 'VRACEN_NA_KOREKCIJU', 'ODOBREN', 'ODBIJEN'),
    putanjaFajla VARCHAR(500),      -- Putanja do skenirane slike obrasca
    ocrIzvrseno BOOLEAN DEFAULT FALSE,
    brojUspesnihStavki INT DEFAULT 0,
    brojNeuspesnihStavki INT DEFAULT 0,
    idZaposlenog INT,               -- FK -> zaposlenifakulteta
    indeks VARCHAR(20)              -- FK -> student (format: "2023/0342")
);
```

### 2.2 Tabela: `stavkeobrasca`

Svaka stavka predstavlja jedno polje iz obrasca koje je OCR obradio.

```sql
CREATE TABLE stavkeobrasca (
    idStavke INT PRIMARY KEY AUTO_INCREMENT,
    idObrazac INT,                  -- FK -> sv20obrazac
    ocrVrednost VARCHAR(500),       -- Vrednost koju je OCR prepoznao
    korigovanaVrednost VARCHAR(500),-- Korigovana vrednost (ako korisnik ispravi)
    nivoPodudarnosti DOUBLE,        -- Confidence score (0.0 - 100.0)
    ocrUspesno BOOLEAN,             -- Da li je OCR uspešno prepoznao
    idPolja INT                     -- FK -> tippolja
);
```

### 2.3 Tabela: `tippolja`

Definiše tipove polja na SV-20 obrascu sa koordinatama za OCR.

```sql
CREATE TABLE tippolja (
    idPolja INT PRIMARY KEY AUTO_INCREMENT,
    nazivPolja VARCHAR(100),        -- Npr: "ime_studenta", "jmbg", "pol"
    tipPodatka ENUM('TEXT', 'NUMERIC', 'DATE', 'BOOLEAN', 'ALPHANUMERIC'),
    regexValidacija VARCHAR(255),   -- Regex za validaciju (npr. za JMBG)
    pozicijaX INT,                  -- X koordinata na slici (pikseli)
    pozicijaY INT,                  -- Y koordinata na slici (pikseli)
    sirina INT,                     -- Širina ROI oblasti
    visina INT,                     -- Visina ROI oblasti
    stranica INT DEFAULT 1,         -- Broj stranice (ako ima više)
    redosledObrade INT,             -- Redosled obrade polja
    podrzavaOCR BOOLEAN DEFAULT TRUE,
    obaveznoPolje BOOLEAN DEFAULT FALSE
);
```

---

## 3. Java Domenski Objekti

### 3.1 SV20Obrazac.java (relevantni delovi)

```java
public class SV20Obrazac {
    private int idObrazac;
    private Date datumUnosa;
    private int skolskaGodina;
    private int semestar;
    private Status status;              // Enum: PODNET, U_OBRADI, ODOBREN...
    private String putanjaDoFajla;      // Putanja do skenirane slike
    private boolean ocrIzvrseno;
    private int brojUspesnihStavki;
    private int brojNeuspesnihStavki;
    private ZaposleniFakulteta idZaposlenog;
    private Student indeks;
    private List<StavkeObrasca> stavke;
}
```

### 3.2 StavkeObrasca.java

```java
public class StavkeObrasca {
    private int idStavke;
    private SV20Obrazac idObrazac;
    private String ocrVrednost;         // Prepoznata vrednost
    private String korigovanaVrednost;  // Korigovana od strane korisnika
    private double nivoPodudarnosti;    // 0.0 - 100.0 (confidence %)
    private boolean ocrUspesno;
    private TipPolja idPolja;
}
```

### 3.3 TipPolja.java

```java
public class TipPolja {
    private int idPolja;
    private String nazivPolja;          // "ime_studenta", "jmbg", "pol"...
    private tipPodatka tipPodatka;      // Enum: TEXT, NUMERIC, DATE, BOOLEAN, ALPHANUMERIC
    private String regexValidacija;     // Npr. "^[0-9]{13}$" za JMBG
    private int pozicijaX;              // X koordinata (pikseli)
    private int pozicijaY;              // Y koordinata (pikseli)
    private int sirina;                 // Širina regiona
    private int visina;                 // Visina regiona
    private int stranica;               // Broj stranice
    private int redosledObrade;         // Redosled OCR obrade
    private boolean podrzavaOCR;
    private boolean obaveznoPolje;
}
```

---

## 4. Zahtevi za Python OCR Servis

### 4.1 Funkcionalni Zahtevi

1. **REST API** na portu `9001` sa sledećim endpointima:
   - `GET /api/health` - Health check
   - `GET /api/tippolja` - Vraća listu tipova polja iz baze (za template)
   - `POST /api/ocr/process` - Prima sliku, vraća JSON sa prepoznatim poljima

2. **OCR Engine**: Koristi **EasyOCR** sa podrškom za:
   - Srpsku ćirilicu (`sr_cyrl`)
   - Srpsku latinicu (`sr_latn`)

3. **Image Processing** sa OpenCV:
   - Deskew (ispravljanje nagiba)
   - Denoise (uklanjanje šuma)
   - Binarization (priprema za OCR)
   - ROI cropping (isecanje regiona po koordinatama iz `tippolja`)
   - Uklanjanje linija (horizontalnih linija za pisanje)
   - Uklanjanje kućica (comb-boxes za JMBG polja)

4. **OMR (Optical Mark Recognition)** za:
   - Checkboxove (Da/Ne)
   - Zaokružene opcije (Pol: M/Ž, Semestar: Zimski/Letnji)

5. **Validacija** ekstrahovanih vrednosti:
   - JMBG: 13 cifara + kontrolna cifra (modul 11)
   - Datum: DD.MM.YYYY format
   - Indeks: GGGG/BBBB format (npr. 2023/0342)

6. **Offline rad**: Mora raditi bez interneta (modeli unapred preuzeti)

### 4.2 Nefunkcionalni Zahtevi

1. **GPU Podrška**: AMD Radeon preko `onnxruntime-directml`
2. **Fallback**: Ako nema GPU, koristi CPU
3. **Timeout**: Max 60 sekundi po slici
4. **Format slike**: JPG, PNG, PDF

---

## 5. Očekivani API Contract

### 5.1 Request: `POST /api/ocr/process`

```
Content-Type: multipart/form-data

file: <binary image data>
obrazac_id: 123  (opciono)
```

### 5.2 Response: JSON

```json
{
  "success": true,
  "message": "Obrada završena",
  "obrazac_id": 123,
  "total_fields": 12,
  "successful_fields": 10,
  "failed_fields": 2,
  "fields": [
    {
      "field_id": 1,
      "field_name": "ime_studenta",
      "field_type": "TEXT",
      "ocr_value": "Вукашин",
      "validated_value": "Вукашин",
      "confidence": 94.5,
      "is_valid": true,
      "validation_error": null,
      "coordinates": {
        "x": 150,
        "y": 320,
        "width": 400,
        "height": 50
      }
    },
    {
      "field_id": 3,
      "field_name": "jmbg",
      "field_type": "NUMERIC",
      "ocr_value": "0101998710029",
      "validated_value": "0101998710029",
      "confidence": 89.2,
      "is_valid": true,
      "validation_error": null,
      "coordinates": {
        "x": 150,
        "y": 450,
        "width": 520,
        "height": 60
      }
    },
    {
      "field_id": 5,
      "field_name": "pol",
      "field_type": "BOOLEAN",
      "ocr_value": "M",
      "validated_value": "M",
      "confidence": 92.0,
      "is_valid": true,
      "validation_error": null,
      "omr_detected": true
    }
  ]
}
```

### 5.3 Mapiranje na Java objekte

Java klijent će parsirati ovaj JSON i kreirati `StavkeObrasca` objekte:

```
JSON field -> StavkeObrasca
─────────────────────────────────────────
field_id        -> idPolja (TipPolja.idPolja)
ocr_value       -> ocrVrednost
validated_value -> korigovanaVrednost (inicijalno isto kao ocr_value)
confidence      -> nivoPodudarnosti
is_valid        -> ocrUspesno
```

---

## 6. Preporučena Struktura Projekta

```
sv20-ocr-service/
│
├── main.py                      # Entry point
├── config.py                    # Konfiguracija
├── requirements.txt             # Dependencies
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
│   ├── omr_processor.py         # OMR za checkboxove
│   └── validators.py            # JMBG, datum, indeks validacija
│
├── database/
│   ├── __init__.py
│   └── db_connector.py          # MySQL konekcija za čitanje tippolja
│
├── models/                      # EasyOCR modeli (offline)
│   ├── craft_mlt_25k.pth
│   ├── latin_g2.pth
│   └── cyrillic_g2.pth
│
├── templates/
│   └── sv20_template.json       # Alternativa bazi - statički template
│
└── tests/
    ├── test_ocr.py
    └── sample_images/
```

---

## 7. Integracija sa Java Sistemom

### 7.1 Tok podataka

```
1. Korisnik uploaduje sliku SV-20 obrasca u Java klijentu
2. Java čuva sliku na disk, kreira SV20Obrazac sa putanjaDoFajla
3. Korisnik klikne "Pokreni OCR"
4. Java šalje HTTP POST na Python servis sa slikom
5. Python:
   a) Učitava tippolja iz baze (ili template JSON)
   b) Za svako polje: crop ROI -> preprocess -> OCR/OMR -> validate
   c) Vraća JSON sa rezultatima
6. Java parsira JSON, kreira StavkeObrasca objekte
7. Java čuva stavke u bazu
8. Korisnik može pregledati/korigovati vrednosti u GUI-ju
```

### 7.2 Konekcija na MySQL bazu

Python servis treba da može da čita `tippolja` tabelu:

```python
# config.py
DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "database": "bazaocr",
    "user": "root",
    "password": ""
}
```

Alternativno, koordinate polja mogu biti u statičkom JSON fajlu ako ne želiš MySQL konekciju.

---

## 8. Specijalni Slučajevi

### 8.1 JMBG Polje (Comb-Box)

JMBG polje ima 13 kućica (comb-boxes). Potrebno je:
1. Ukloniti vertikalne linije kućica
2. Prepoznati samo cifre
3. Validirati kontrolnu cifru (modul 11)

### 8.2 OMR Polja (Checkboxovi/Zaokruženo)

Za polja kao "Pol" (M/Ž) ili "Semestar" (Zimski/Letnji):
1. Detektovati koja opcija ima više crnog piksela (fill ratio)
2. Ili detektovati krug oko teksta

### 8.3 Rukopis vs. Štampano

- Obrazac može imati kombinaciju štampanog teksta i rukopisa
- EasyOCR podržava oboje, ali rukopis ima niži confidence

---

## 9. Prioriteti Implementacije

1. **Faza 1**: Basic OCR
   - Health endpoint
   - Učitavanje slike
   - Jednostavan OCR celog dokumenta
   - JSON response

2. **Faza 2**: Template-based OCR
   - Čitanje tippolja iz baze/JSON
   - ROI cropping po koordinatama
   - Preprocessing per-field

3. **Faza 3**: Validacija
   - JMBG validacija
   - Datum parsing
   - Indeks format

4. **Faza 4**: OMR
   - Checkbox detekcija
   - Zaokružene opcije

5. **Faza 5**: Optimizacija
   - GPU akceleracija
   - Caching
   - Error handling

---

## 10. Test Podaci

### Primer tippolja zapisa:

| idPolja | nazivPolja | tipPodatka | regexValidacija | pozicijaX | pozicijaY | sirina | visina |
|---------|------------|------------|-----------------|-----------|-----------|--------|--------|
| 1 | ime_studenta | TEXT | NULL | 150 | 320 | 400 | 50 |
| 2 | prezime_studenta | TEXT | NULL | 150 | 380 | 400 | 50 |
| 3 | jmbg | NUMERIC | ^[0-9]{13}$ | 150 | 450 | 520 | 60 |
| 4 | broj_indeksa | ALPHANUMERIC | ^\d{4}/\d{4}$ | 150 | 520 | 300 | 50 |
| 5 | pol | BOOLEAN | NULL | 600 | 320 | 200 | 60 |
| 6 | datum_rodjenja | DATE | NULL | 150 | 590 | 250 | 50 |

---

## 11. Očekivani Output

Kada završiš implementaciju, očekujem:

1. **Funkcionalan Python servis** koji se pokreće sa `python main.py`
2. **REST API** na portu 9001
3. **Dokumentacija** za pokretanje i konfiguraciju
4. **requirements.txt** sa svim dependencies
5. **download_models.py** za offline pripremu

---

## 12. Napomene

- **Bez eksternih API poziva**: Sve mora raditi offline
- **AMD GPU**: Korisnik ima AMD Radeon, koristi `onnxruntime-directml`
- **Srpski jezik**: Prioritet je srpska ćirilica i latinica
- **Seminarski rad**: Ovo je projekat za fakultet, treba biti robustan ali ne preterano kompleksan

---

**Autor konteksta**: Vukasin Lukic
**Projekat**: Seminarski rad - Softversko inženjerstvo
**Fakultet**: Fakultet organizacionih nauka, Beograd
