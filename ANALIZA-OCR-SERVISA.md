# Analiza SV-20 OCR mikroservisa (avgust 2026)

> Napomena: ovo je čisto analitički dokument. Ništa u kodu nije menjano.
> Zasnovano na stvarnom kodu u `sv20-ocr-service/` (ne na CLAUDE.md planu, koji je
> originalna specifikacija — stvarna implementacija je otišla dosta dalje i drugačije
> od nje). Kao dokaz koristim i stvarne rezultate iz `ocr_result_SMART_CLEAN.json`.

---

## 1. Šta je ovo zapravo (stanje projekta)

Ovo **nije** mala seminarska skripta — to je ozbiljno razvijen servis:

| Fajl | Linija koda | Šta radi |
|---|---|---|
| `api/server.py` | 1114 | FastAPI endpoints, orkestracija OCR/OMR obrade, template editor API |
| `processors/image_processor.py` | 595 | OpenCV pipeline: deskew, denoise, comb-box removal, perspective transform, PDF→slika |
| `processors/omr_logic.py` | 873 | 4 različite heuristike za detekciju zaokruženih opcija |
| `processors/ocr_engine.py` | 374 | EasyOCR wrapper |
| `processors/validators.py` | 662 | JMBG/datum/indeks validacija + "smart cleaning" OCR grešaka |
| `templates/sv20_template.json` | 1218 | koordinate za **37 polja** na 2 strane obrasca |
| `static/index.html` | 1126 | vizuelni **Template Editor** (browser alat za crtanje ROI pravougaonika) |

Za poređenje, CLAUDE.md plan opisuje ~12 polja i jednostavan pipeline. Stvarni kod pokriva
kompletan ŠV-20 obrazac (37 polja, 2 strane, mešavina teksta/rukopisa/zaokruživanja/potpisa)
i ima čitav vizuelni alat za kalibraciju koordinata — to je solidan inženjerski rad, samo
nedovoljno testiran/dokumentovan u odnosu na svoju veličinu. Nema `tests/` foldera uopšte,
iako ga i README i masterPromtPlan.md pominju.

---

## 2. Kako Java aplikacija koristi mikroservis

**Bitna napomena:** u ovom repozitorijumu **ne postoji Java kod** (KLIJENT/SERVER moduli
su, po `masterPromtPlan.md`, poseban projekat). Ono što sledi je dokumentovani ugovor
(contract) između sistema, rekonstruisan iz `README.md`, `masterPromtPlan.md` i stvarnog
JSON odgovora servera — ne mogu da garantujem da Java strana tačno tako radi, samo da je
tako **specificirano**.

### Tok podataka

```
1. Korisnik u Java Swing KLIJENT-u učita sken/PDF ŠV-20 obrasca
2. Java SERVER sačuva fajl na disk, kreira red u `sv20obrazac` (MySQL)
3. Korisnik klikne "Pokreni OCR"
4. Java šalje HTTP POST multipart/form-data na http://127.0.0.1:9001/api/ocr/process
   (polja: file, obrazac_id, page_number)
5. Python servis:
   - učita templates/sv20_template.json (statički, NE iz baze — vidi odeljak 4.7)
   - za PDF: konvertuje SVE strane u slike (PyMuPDF), obrađuje sve ili samo `page_number`
   - resize na 2048px širine, preprocesuje (perspective transform, deskew, denoise)
   - za svako polje: crop ROI → OCR (EasyOCR) ili OMR (OpenCV konture) → validacija
   - vraća JSON: { total_fields, successful_fields, failed_fields, fields: [...] }
6. Java parsira JSON i pravi `StavkeObrasca` objekte, jedan po polju:

     JSON polje        → Java/DB kolona
     ─────────────────────────────────────────
     field_id           → idPolja (FK na tippolja)
     ocr_value           → ocrVrednost
     validated_value     → korigovanaVrednost (inicijalno)
     confidence (0–100)  → nivoPodudarnosti
     is_valid             → ocrUspesno

7. Java upisuje stavke u `stavkeobrasca` tabelu
8. Korisnik u GUI-ju pregleda/ispravlja vrednosti (posebno one sa niskim confidence-om
   ili is_valid=false)
```

### Ostali endpoint-i koje Java (ili operator) može zvati

- `GET /api/health` — da li je servis gore, da li je GPU/OCR inicijalizovan
- `GET /api/template` / `GET /api/fields` — lista polja (za dinamičko renderovanje forme)
- `POST /api/template/save` — čuvanje izmenjenog template-a sa Template Editora
- `POST /api/ocr/process-raw` — OCR bez template-a (sirovi tekst blokovi, za debug)
- `POST /api/debug/omr-rois` — vraća isečene OMR regione kao base64 slike + fill_ratio
  po opciji (koristio si ovo dok si kalibrisao `bracni_status` i `vrsta_studija` polja —
  otud `debug_*_annotated.png` fajlovi u repou)

### Bitna neusklađenost dokumentacije

`CLAUDE.md` (originalni plan) pokazuje `confidence: 0.94` (opseg 0–1), dok stvarna baza
(`masterPromtPlan.md`, `nivoPodudarnosti DOUBLE`) i stvarni kod (`confidence * 100` u
`server.py:688`) koriste opseg **0–100**. Kod je usklađen sa bazom — CLAUDE.md je samo
zastareo plan. Ako iko čita CLAUDE.md kao referencu za Java stranu, ovo će zbuniti.

---

## 3. Šta je DOBRO urađeno (da ne bude sve kritika)

1. **"Smart cleaning" heuristike u `validators.py`** (npr. `clean_year_ocr`,
   `clean_date_ocr`) rešavaju stvarne, specifične OCR greške koje si očigledno video
   uživo (npr. "2925"→"2025", "12101006"→godina). Ovo nije generički kod prepisan iz
   tutorijala — vidi se da je tuning rađen na pravim skeniranim primerima.
2. **Comb-box removal za JMBG** (`image_processor.py:221-279`) je prošao kroz više
   iteracija (komentari "Aggressive mode", biranje cross-kernela da ne pojede "1" i "7")
   — pravi trag debagovanja na realnim slikama.
3. **Precizne-koordinate OMR metoda** (`_detect_by_precise_coords`) u kombinaciji sa
   Template Editorom (`static/index.html`) je pragmatično i pametno rešenje: umesto da
   nagađaš geometriju, ti ručno nacrtaš pravougaonik oko svakog kružića u browseru i
   sačuvaš koordinate. Za jedan fiksni obrazac (uvek isti raspored) ovo realno radi
   bolje od "pametnijih" opštih OMR algoritama.
4. **PDF multi-page podrška** i **2x upscaling za OCR dok se koordinate drže na 1024px
   bazi** (`server.py:434-437`) — razuman kompromis: editor ostaje brz i lagan, a OCR
   ipak radi na rezoluciji dovoljnoj za sitan rukopis.
5. **Response format** se lepo poklapa sa DB šemom iz `masterPromtPlan.md` — mapiranje
   `field_id → idPolja`, `confidence → nivoPodudarnosti` je 1:1, nema nepotrebnog
   prevođenja na Java strani.
6. Servis je vezan na `127.0.0.1` (nije izložen mreži) — u skladu sa "offline" zahtevom.

---

## 4. Stvarne mane (rangirano po ozbiljnosti)

### 4.1 🔴 JMBG validacija tiho PREPISUJE poslednju cifru — lažna sigurnost

`validators.py:141-164`, `clean_jmbg_ocr(..., auto_correct_checksum=True)`:

Kad kontrolna cifra ne odgovara, kod **ne javlja grešku** — on **izračuna "ispravnu"
cifru i zameni je**, pa vrati `is_valid=True`. Dokaz iz tvog stvarnog rezultata
(`ocr_result_SMART_CLEAN.json`, polje `jmbg`):

```json
"ocr_value": "0101006710111", "validated_value": "0101006710116",
"confidence": 98.08, "is_valid": true
```

OCR je pročitao `...111`, sistem je preko modula 11 izračunao da treba `...116` i **prosto
zamenio poslednju cifru** — bez ikakvog upozorenja da se nešto promenilo. Problem: kontrolna
cifra u JMBG-u postoji da otkrije grešku, ne da bude "auto-fix dugme". Ako je OCR pogrešio
bilo koju od **prvih 12** cifara (dan, mesec, godina, region, redni broj), checksum i dalje
može da "prođe" nakon što se poslednja cifra izmisli da odgovara — validacija onda kaže
`is_valid: true, confidence: 98%` za JMBG koji realno pripada nekom drugom studentu ili je
potpuno pogrešan datum rođenja/pol. Ovo je najozbiljniji nalaz jer JMBG ide direktno u bazu
fakulteta i može da poveže obrazac sa pogrešnim studentom, a operater u GUI-ju nema razloga
da posumnja jer piše "validno, 98%".

**Predlog:** checksum treba SAMO da validira (report mismatch → `is_valid=false`,
`validation_error="kontrolna cifra ne odgovara"`), nikad da tiho menja OCR vrednost.
Auto-correct opcija može ostati, ali mora da se vrati kao **odvojeno polje**
(`auto_corrected: true`, `original_ocr_value`, `corrected_value`) tako da Java/GUI eksplicitno
prikaže "sistem je promenio poslednju cifru — proveri ručno", a ne da to sakrije iza `is_valid=true`.

### 4.2 🔴 Ćirilica i latinica se NE čitaju istovremeno, iako to piše svuda u dokumentaciji

`config.py:26-30` deklariše `OCR_LANGUAGES = ['rs_cyrillic', 'rs_latin', 'en']`, ali
`ocr_engine.py:38-44`:

```python
# Fix for "Cyrillic is only compatible with English" error
if 'rs_cyrillic' in languages:
     self.languages = ["ru", "rs_cyrillic", "be", "bg", "uk", "mn", "en"]
```

Ovo je legitiman workaround za stvarno EasyOCR ograničenje (ćirilični i latinični modeli
zaista ne mogu u isti `Reader` — grupe karaktera se ne poklapaju), ali **posledica je da se
`rs_latin` NIKAD ne učitava**, iako je default konfiguracija uvek sa `rs_cyrillic` u listi.
Pošto ŠV-20 obrazac ima polja koja se često popunjavaju latinicom (imena poput "Vukašin",
mešano rukom pisan tekst), ta polja se trenutno čitaju kroz model treniran za ruski/bugarski/
ukrajinski/mongolski ćirilični skup — latinična slova prepoznaje samo "slučajno", preko engleskog
dela skupa. Vidljivo u tvom test rezultatu: polje `vrsta_studija` vraća `"Osnovne akademske
studije"` (latinica) — ali to je OMR vrednost iz template-a, ne OCR pročitan tekst, pa se
problem ne vidi tamo. Realan test bi bio neko čisto rukom pisano latinično polje.

**Predlog:** dva odvojena `EasyOCR.Reader` procesa (jedan cirilični, jedan latinični), pa ili
(a) označiti u template-u očekivani pismo po polju (najjeftinije i najpouzdanije — ti već znaš
da je "univerzitet" štampan ćirilicom a potpis može biti bilo šta), ili (b) pokrenuti oba i
zadržati rezultat sa višim confidence-om. Ovo udvostručava RAM/vreme inicijalizacije, ali
polja se obrađuju sekvencijalno pa dodatni trošak po requestu nije velik.



#Ja zelim da uradimo opciiju b da pokrenemo oba i zadrzimo rezultat sa visim confidencom ... 

### 4.3 🟠 "AMD GPU akceleracija" verovatno uopšte ne radi

Ceo GPU sloj (`config.py:33-35`, `Config.detect_gpu()`, `ocr_engine.py:_detect_hardware`)
proverava da li **`onnxruntime`** vidi `DmlExecutionProvider`, pa na osnovu toga zove
`easyocr.Reader(gpu=True)`. Problem: **EasyOCR interno koristi PyTorch**, ne
`onnxruntime` — GPU odluku unutar `torch` donosi `torch.cuda.is_available()`, koji na AMD
GPU-u pod Windows-om (bez ROCm-a, koji ionako ne postoji za Windows) vraća `False`.
`onnxruntime-directml` je potpuno nepovezan paket od onoga što EasyOCR stvarno koristi za
inferencu — instaliranje `onnxruntime-directml` (kako i piše u `requirements.txt`) ne daje
EasyOCR-u nikakvo ubrzanje. Realno, servis po svoj prilici **stalno radi na CPU-u**, bez obzira
šta `gpu_backend` polje u `/api/health` javlja ("directml" je detektovan onnxruntime provider,
ne EasyOCR backend). Ovo objašnjava i vremena obrade koja si video (`processing_time_ms:
17322` za jednu stranu u primeru iznad — ~17 sekundi je tipično za CPU inferencu na 37 polja,
ne za GPU).

Dodatno, sam Microsoft trenutno (2026) vodi DirectML repo kao **"in maintenance mode"** —
tj. ne razvija se dalje aktivno, pa i da se poveže ispravno, to nije pravac u koji vredi
dodatno ulagati na duže staze.

**Predlog:** ako ti realno treba GPU ubrzanje na AMD kartici, dve opcije:
- `torch-directml` paket (pravi PyTorch DirectML plugin) + eksplicitno stavljanje tenzora na
  `dml` device — zahteva sitne izmene u EasyOCR pozivnom kodu jer EasyOCR ne podržava custom
  torch device odmah "od kutije".
- Ili pređi na OCR engine koji **stvarno** radi kroz ONNX (npr. **OnnxTR**, ONNX verzija
  docTR-a) — tu `onnxruntime-directml` koji već imaš u `requirements.txt` konačno postaje
  koristan, jer ONNX runtime zaista pokreće inferencu.
- Za seminarski rad — najjeftinije rešenje je samo iskreno reći "radi na CPU, GPU je out of
  scope" i izbaciti pogrešnu DirectML priču iz prezentacije/README-a.

### 4.4 🟠 Nema pravog anchor-based poravnanja — tačno ono što je originalno traženo, a nije implementirano

U tvom originalnom zahtevu (CLAUDE.md, sekcija "Anchor-Based Alignment") tražio si
template matching na logo "РЗС" i naslov "Образац ШВ-20", da koordinate polja budu
**relativne** u odnosu na ta sidra. Kod za to postoji —
`image_processor.py:375-401`, `detect_anchor()` — ali se **nigde ne poziva**. Isto tako,
`sv20_template.json` ima `document.anchor_elements` sekciju sa pozicijama loga/naslova —
takođe se nigde ne čita, čisto mrtvi podaci u JSON-u.

Ono što se stvarno dešava u pipeline-u (`preprocess_full_document`,
`image_processor.py:490-510`) je generički `perspective_transform()` koji traži bilo koji
četvorougaoni kontur koji pokriva >50% slike (ivice papira), plus `deskew()` preko Hough
linija. Ovo pomaže kod rotacije/perspektive celog dokumenta, ali **ne pomaže ako je sken
pomeren, opsečen drugačije, ili skeniran na skeneru sa drugom DPI kalibracijom** — sve
koordinate u template-u su **apsolutni pikseli** vezani za JEDNU referentnu sliku širine
1024px sa kojom si kalibrisao u Editoru. Ako sledeći sken ima i minimalno drugačiju marginu
(npr. drugi skener, drugačije umetnut papir), SVA polja se pomeraju zajedno i ROI-jevi seku
pogrešan deo forme — nema mehanizma koji to detektuje ili kompenzuje.

**Predlog:** iskoristi `detect_anchor()` koji već postoji — pronađi 2 stabilne referentne
tačke (npr. sam okvir tabele ili "РЗС" logo), izračunaj offset (dx, dy) i eventualno scale
u odnosu na referentnu poziciju iz template-a, pa transformiši SVE koordinate polja tim
offsetom pre cropovanja. Ovo je nekoliko sati posla jer je 80% infrastrukture (funkcija,
template polje za anchor poziciju) već tu — samo nije povezano.


#Ovo moze da se uradi i da se uradi toogle na onom sajtu gde rucno trazim kordinate a li da mi nikako ne obrises kod za rucno postavljanje kordinata i celu tu logiku ili da je ne koristis nego je samo stavi pod neko dugme i da ja radim to tugle dugme tu i da biram nacin na koji zelim da radim razvijamo i ovaj tvoj ancor nacin jer je bolji ali cuvamo ovaj koji trenutno radi za fallback ,

### 4.5 🟡 Confidence skorovi za OMR polja su delimično izmišljeni brojevi

U `omr_logic.py` postoji dosta mesta gde confidence nije stvarno izračunat nego je
**fiksna konstanta**:

```python
confidence = 0.8 if detected_values else 0.0        # linija 729 (detect_multi_select fallback)
confidence = 0.9 if detected_values else 0.0        # linija 800 (_detect_multi_by_precise_coords)
confidence = 0.7 if max_score > 0.1 else 0.5         # linija 424
```

U tvom test rezultatu ovo se lepo vidi: `pol` polje ima `confidence: 70.0` (tačno),
`izdrzavanje_drugih` ima `70.0`, `skolska_sprema_majke` ima `70.0`, `nacin_finansiranja`
ima `100.0`, `potrebna_podrska` ima `90.0` — previše "okruglih" brojeva da bi bili stvarno
izmereni signal. Problem: Java GUI (po specifikaciji) treba da koristi `nivoPodudarnosti`
da signalizira operateru koja polja da ručno proveri. Ako je 70.0 hardkodovana konstanta i
za pouzdano i za granično detektovanu opciju, taj signal je beskoristan baš tamo gde je
najpotrebniji — kod OMR polja gde je greška najskuplja (npr. pogrešno pročitan "pol" ili
"način finansiranja").

**Predlog:** confidence uvek računaj iz stvarne razlike između najboljeg i drugog najboljeg
kandidata (margin), normalizovano na 0–1, kao što `_detect_by_precise_coords` već radi na
nekim mestima — samo primeni isti princip svuda, uključujući multi-select granu.

molim te ovu logiku jos vise istrazi na webu i uradi to ne zelim nista hardkodovano i izmislejno ... 

### 4.6 🟡 4 sloja OMR heuristika sa desetinama "magic number" pragova

`detect_marked_option` prolazi kroz: precizne koordinate → kontura-krug → edge density →
zone density, svaki sa svojim pragovima (`0.6`, `0.5`, `0.3`, `0.15`, `0.10`, `0.05`...).
Ovo radi (test na dnu `omr_logic.py` prolazi), ali je krhko za održavanje — svaki prag je
"nameštan okom" na par test slika, bez ikakvog automatizovanog regresionog testa koji bi
upozorio ako izmena jednog praga pokvari drugo polje. Realno, pošto sada VEĆ imaš precizne
koordinate iz Template Editora za sve opcije (has_precise_coords je skoro uvek True), tri
fallback metode se u produkciji verovatno retko ili nikad ne pozivaju — mrtav kod koji
otežava čitanje.

**Predlog:** ako je Editor workflow (ručno kalibrisane koordinate) stalni način rada, smelo
obriši tri fallback metode i zadrži samo `_detect_by_precise_coords` — manje koda, lakše
za razumeti i testirati.

### 4.7 🟡 Nema veze sa bazom, template je samo static JSON

`masterPromtPlan.md` opisuje da servis treba da čita `tippolja` tabelu iz MySQL-a
(`DB_CONFIG` postoji i u `config.py`, ali se nigde ne koristi). U praksi je to u redu za
seminarski rad (JSON je jednostavniji i dovoljno dobar), ali znači da ako neko na Java
strani promeni/doda polje u bazi (`tippolja`), OCR servis to neće videti dok se ručno ne
prepiše i u `sv20_template.json` preko Template Editora. Dva izvora istine za isti podatak
— vredi barem napomenuti u README-u da je JSON zvanični izvor, ne baza, da se ne bi neko
kasnije zbunio zašto izmena u bazi "ne radi".

### 4.8 🟢 Nema automatizovanih testova

`tests/` folder ne postoji, iako ga i `README.md` i `masterPromtPlan.md` pominju
(`pytest tests/`). Svi debug fajlovi u root-u (`debug_result*.json`,
`ocr_result_*.json`) su ručni jednokratni eksperimenti, ne ponovljivi testovi. Za odbranu
seminarskog ovo nije nužno kritično, ali čak i 2-3 `pytest` smoke testa (npr. "JMBG checksum
validacija tačno odbija loš broj", "poznata slika daje očekivan `broj_indeksa`") bi bili
jeftina odbrana protiv regresije dok tuning-uješ pragove pred odbranu.

### 4.9 🟢 Sitnice

- `app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, ...)` —
  kombinacija wildcard origin + credentials je nešto što moderni browseri odbijaju po
  specifikaciji; pošto Java klijent nije browser, ovo trenutno ne pravi problem, ali ako
  ikad neko doda web dashboard koji čita kolačiće/auth, ovo će prvo pući. Bezopasno da se
  ostavi za sada.
- Endpoint-i nemaju nikakvu autentikaciju (uključujući `POST /api/template/save`, koji
  prepisuje template fajl). Za lokalni offline seminarski rad na `127.0.0.1` je prihvatljivo,
  samo ne kopirati ovaj obrazac 1:1 u nešto što ide na mrežu.
- Dosta `logger.info` poziva ostavljeno je na produkcionom putu obrade (posebno u
  `/api/debug/omr-rois` i `process_omr_field`) — funkcionalno bezopasno, ali čini logove
  glomaznim; vredi razdvojiti DEBUG od INFO nivoa pre odbrane da log ne bude 500 linija
  po jednoj slici.

---

## 5. Šta 2026 nudi kao alternative/dopune (kratak pregled)

| Tehnologija | Za šta bi pomogla ovde | Napomena |
|---|---|---|
| **TrOCR** (transformer OCR) | Rukom pisana polja (ime, adresa, mesto rođenja) — literatura ga navodi kao najbolji izbor baš za rukopis, bolji od EasyOCR-a na tom zadatku | Sporiji je i teži (transformer), realno kao "drugo mišljenje" za polja sa niskim EasyOCR confidence-om, ne kao potpuna zamena |

obavezno .... nam treba rukom pisanja polja da citamo !!!!


| **OnnxTR / docTR (ONNX)** | Pravi put do stvarne AMD GPU akceleracije preko `onnxruntime-directml` koji već imaš u `requirements.txt` | Za razliku od EasyOCR-a, ovaj pipeline stvarno prolazi kroz ONNX Runtime, pa DirectML provider ovde ima efekta |



| **torch-directml** | Alternativa ako želiš da ostaneš na EasyOCR-u a stvarno dobiješ GPU | Zahteva eksplicitno device-mapping izmene u pozivnom kodu; Microsoft repo je u "maintenance mode" pa nije dugoročno najsigurnija opklada |
| **PaddleOCR** | Bolji layout-analysis za tabelarne/strukturirane delove forme | Manje poznat po ćirilici/srpskom nego EasyOCR, treba testirati kvalitet pre prelaska |
| **Fiducijalna sidra (corner marks)** umesto generičkog "detect 4 corners of paper" | Rešava tačno problem iz 4.4 — industrija (npr. Remark OMR) standardno stavlja tamne kvadrate u sva 4 ugla forme baš zbog pouzdanog poravnanja bez obzira na skew/pomeraj | Zahtevalo bi da se u dizajn samog obrasca (ako ikad štampaš svoju verziju) dodaju mali crni kvadrati u uglove; ako moraš da radiš sa postojećim državnim ŠV-20 obrascem, ostaje ti aktiviranje već napisanog `detect_anchor()` na logo/naslov teksta |
Da li mi mozemo na slici da dodamo conrer marks da se nista ne mora menjati i onda da radimo na osnovu toga ... 




**Realan predlog prioriteta** (od najjeftinijeg/najvrednijeg ka opcionom):
1. Ispraviti JMBG auto-correct da ne laže o validnosti (4.1) — pola dana posla, sprečava, obavezno i istrazi web kako se radi tacan jmbg validacija ... 


   pogrešno povezivanje obrazaca sa studentima.
2. Popraviti/objasniti GPU priču — ili stvarno povezati `torch-directml`, ili iskreno
   napisati u dokumentaciji da trenutno radi na CPU-u (4.3) — pola dana.
3. Povezati `detect_anchor()` u pipeline (4.4) — najveći uticaj na robusnost prema
   varijacijama u skeniranju, ali i najviše posla (dan-dva).
4. Realan dual-script OCR (4.2) — direktno utiče na tačnost teksta, srednje velik posao
   (novi Reader instance + logika biranja rezultata).
5. Confidence bez magic numbers (4.5) — čini "koje polje treba ručno proveriti" signal
   pouzdanim, pola dana.
6. Skinuti mrtve OMR fallback grane (4.6) i dodati par pytest testova (4.8) — čisto
   održavanje, uradi kad imaš vremena, nije hitno pred odbranu.


Moramo obavezno ispostovati ovaj relacioni model na kom radi java aplikacija ... 

tako da moram oda se prilagodimo da ce ona da salje zapravo  nesto od podataka odavde sto ima smisla 3.5 Структура софтверског система – Релациони модел
На основу концептуалног модела се прави релациони модел.
1.ZaposleniFakulteta (idZaposlenog, ime, prezime, korisnickoIme, email, sifra)
2.TipPolja(idPolja, nazivPolja, tipPodatka, regexValidacija, pozicijaX, pozicijaY, sirina,
visina, stranica, redosledObrade, podrzavaOCR, obaveznoPolje)
3.StudijskiProgram(idStudProgram, naziv, oznaka, stepenStudija)
4.TerminDezurstva(idTerminDezurstva, tipTermina, kancelarija)
5.Student(indeks, jmbg, ime, prezime, mestoRodjenja, adresaStanovanja, idStudProgram)
6.SV20Obrazac(idObrazac, datumUnosa, skolskaGodina, semestar, status, putanjaFajla,
ocrIzvrseno, brojUspesnihStavki, brojNeuspesnihStavki, idZaposlenog, indeks)
7.StavkeObrasca(idObrazac, idStavke, ocrVrednost, korigovanaVrednost,
nivoPodudarnosti, ocrUspesno, idPolja)
8.Zaposleni-Termin (datum,idZaposlenog, idTerminDezurstva, brojSati, vanredan)  tako da ti skonta sta ce verovatno da salje i sta ce da prima 

---

## 6. Status implementacije (radni paketi iz komentara iznad)

Plan rada je zapisan u `C:\Users\INSOMNIA\.claude\plans\hidden-tinkering-pixel.md` (6 radnih
paketa, redosled od najsigurnijeg ka najrizičnijem). Status:

### ✅ WS1 — JMBG validacija (4.1) — GOTOVO, testirano

`processors/validators.py`: `clean_jmbg_ocr()` više ne dira cifre. `validate_jmbg()` vraća
`(is_valid, cleaned_value, error, suggested_value)` — kad checksum ne prolazi, ostaje
`is_valid=False` i `cleaned_value` je TAČNO ono što je OCR pročitao, bez izmene. Dodata
`suggest_jmbg_correction()` koja predlaže ispravku samo kad je jednoznačna (probom svih 13
pozicija kroz standardne OCR zabune cifara 0↔8, 1↔7 itd. — ako više od jedne zamene daje
validan checksum, ne predlaže ništa). Novo `suggested_value` polje u JSON odgovoru,
additive (Java strana ga može ignorisati bez izmena).

**Potvrđeno na tvom `primer.pdf`** (pravi request kroz `/api/ocr/process`):
```
PRE:  ocr_value="0101006710111" validated_value="0101006710116"  is_valid=true   (LAGALO!)
POSLE: ocr_value="0101006710111" validated_value="0101006710111" is_valid=false  (istina)
```

`sv20-ocr-service/tests/test_validators.py` — 7 testova, svi prolaze.

### ✅ WS2 — Statistička OMR confidence (4.5) — GOTOVO, testirano (uz jednu ozbiljnu korekciju usput)

Prva verzija (medijana/MAD preko SVIH OMR polja na strani, "modified Z-score" iz standardne
statističke literature o outlier-ima) je prošla sve sintetičke testove, ali je na **stvarnom
skenu (`primer.pdf`, preko `/api/debug/omr-rois`) pala** — polja poput "Pol" ili "Način
finansiranja" su prestala da prepoznaju bilo šta. Uzrok: različita OMR polja na ŠV-20 obrascu
imaju drastično različitu "prirodnu" količinu odštampanog mastila (polje sa dugim tekstualnim
opcijama nasuprot polju sa dva mala kružića) — mešanje svih polja u JEDNU statistiku je
mešalo nesamerljive skale. Ovo sam uhvatio testiranjem na tvom fajlu PRE nego što je ušlo u
finalnu verziju — tačno razlog zašto radimo "polako", korak po korak.

Popravljeno: `_relative_margin_confidence()` — razmerna (ratio) mera, RAČUNA SE UNUTAR SAMOG
POLJA (bez mešanja sa drugim poljima): `confidence = (best - reference) / best`. Za
single-select, reference je druga najjača opcija; za multi-select, svaka opcija se poredi sa
najslabijom u tom istom polju. `confidence > 0.5` prirodno znači "pobednik ima duplo više
signala od reference" — simetrična tačka, ne broj biran "na oko".

**Potvrđeno na `primer.pdf`** (pre/posle popravke ovog drugog bug-a):
```
nacin_finansiranja: PRE popravke=NIJE detektovano (conf 5.6%) -> POSLE="Samofinansiranje" (conf 71.1%)
pol:                ostaje nedetektovano (conf 1.3%) - opcije imaju POČTI IDENTIČAN fill
                     (0.2469 vs 0.2503) - iskreno "ne znam", umesto starog izmišljenog "Z, 70%"
```

`sv20-ocr-service/tests/test_omr_logic.py` — 6 testova (uključujući 2 regresiona testa sa
STVARNIM brojevima izmerenim na tvom skenu, da se ova greška ne ponovi), svi prolaze.

**Usput otkriven i popravljen dodatni bag** (`api/server.py`): grana za obradu obične slike
(ne-PDF upload) nikad nije skalirala `omr_options` koordinate na 2x rezoluciju kojom se radi
OCR — samo je PDF grana to radila. To je značilo da OMR polja na direktno upload-ovanim
slikama (JPG/PNG) verovatno uopšte nisu radila ispravno. Sad obe grane skaliraju isto.

### ✅ WS3 — Dual-script OCR, cirilica + latinica (4.2, opcija B) — GOTOVO, testirano

`processors/ocr_engine.py`: `OCREngine` sad drzi DVA EasyOCR readera
(`reader_cyrillic` = `["ru","rs_cyrillic","be","bg","uk","mn","en"]`, `reader_latin` =
`["rs_latin","en"]`). `recognize_single_field()` pokrece OBA nad istim ROI-jem za TEXT/
ALPHANUMERIC polja i zadrzava rezultat sa visim confidence-om (tacno opcija B koju si
trazio). Za cisto numericka polja (JMBG, godine...) koristi se samo cirilicni reader (cifre
nisu pismo-zavisne) da se ne duplira nepotreban trosak. `/api/info` sad iskreno prikazuje
oba jezicka skupa.

**Potvrdjeno na `primer.pdf`**: svih 23 tekstualnih polja je i dalje tacno procitano (ovaj
konkretan test obrazac je gotovo u potpunosti cirilican, ocekivano). Jedan sitan, ocekivan
nusefekat: polje `prebivaliste_naselje` je "Срамчица" (cirilica, conf=0.64) izgubilo od
"Cpemywya" (latinica, conf=0.68, ali besmisleno procitano) - razlika u confidence-u je bila
mala (0.04) i latinicki model je za ovaj konkretan mali/nejasan ROI dao (pogresno) vecu
sigurnost. Ovo je poznat, ocekivan rizik ciste "veci confidence pobedjuje" strategije koji
je i sam plan predvideo - ako se pokaze da smeta u praksi, sledeci jeftin korak bi bio da
template oznaci ocekivano pismo po polju (opcija (a) iz analize) umesto cistog nadmetanja.

**Trosak**: obrada stranice sa primer.pdf porasla je sa ~12.5s na ~19.7s (dual-run se radi
samo za TEXT/ALPHANUMERIC polja, NUMERIC/JMBG/OMR polja nisu pogodjena).

### ✅ WS4 — Anchor-based poravnanje kao toggle (4.4) — GOTOVO, testirano end-to-end

`processors/image_processor.py`: nova `compute_anchor_offset()` iskorišćava postojeći
(ranije nikad pozvan) `detect_anchor()` - za svako sačuvano sidro template-matching
nalazi ga na trenutnoj slici, poredi sa očekivanom pozicijom → translacija (dx, dy) +
uniformni scale (ako su 2 sidra). `api/server.py`: `template.document.alignment_mode`
("manual" default / "anchor") kontroliše da li se offset primenjuje - kad je "manual"
ponašanje je bit-za-bit identično kao pre. `static/index.html`: novi toggle u
sidebaru ("Ručno" / "⚓ Anchor (beta)") + "Uredi sidra" alat koji radi po **istom
obrascu kao postojeće Circle Mode drag/resize** (nova, odvojena logika - ništa od
postojećeg polje/krug koda nije menjano niti obrisano). Sidra se pri Save-u seku iz
referentne slike u browseru i čuvaju kao base64 PNG direktno u template.json.

**Dva prava bug-a nađena i ispravljena tokom testiranja** (ne pretpostavka - uhvaćeno
pisanjem izolovanih testova sa pravim skenom pre integracije):
1. `detect_anchor()` vraća CENTAR poklapanja, a `reference_position` je gornji-levi
   ugao (konvencija koju koristi sve ostalo u kodu) - upoređivanje bez konverzije je
   davalo dx/dy pomeren za pola širine/visine sidra.
2. OpenCV `cv2.matchTemplate` (TM_CCOEFF_NORMED) daje lažno visoku "sličnost" za
   POTPUNO RAVNE (bez teksture) template slike - poznat degenerativni slučaj
   normalizovane korelacije. Dodata zaštita: sidro bez dovoljno kontrasta
   (std < 5) se odbija sa jasnom porukom umesto da tiho izazove lažno poravnanje.

**Potvrđeno na `primer.pdf`** (test skripta koja privremeno menja i vraća template.json):
```
Ista strana, bez veštačkog pomeraja: dx=0.0, dy=0.0, scale=1.000 → alignment_mode_used="anchor"
                                       polja identična manual modu (do 15. decimale confidence-a)
Veštački pomerena slika (+40,+25px):  detektovano dx=40.0, dy=25.0 (tačno pogođeno)
Sidro koje se ne poklapa nigde:       alignment_mode_used="manual_fallback" (graciozan pad,
                                       rezultati i dalje normalno popunjeni, bez crash-a)
```

### ✅ WS6 — TrOCR za rukom pisana polja (eksperimentalno, feature-flag) — GOTOVO, testirano sa pravim modelom

Novi `processors/handwriting_ocr.py` (lenjo učitavanje - `transformers` se uvozi tek
pri prvom pozivu, servis normalno radi bez tog paketa dok je flag isključen).
`Config.ENABLE_HANDWRITING_TROCR = False` (default). Kad je uključeno: za TEXT/
ALPHANUMERIC polja gde je EasyOCR confidence ispod `Config.
HANDWRITING_FALLBACK_CONFIDENCE` (0.5) ili je polje ručno označeno
`"handwriting": true`, pokreće se `cyrillic-trocr/trocr-handwritten-cyrillic` kao
drugo mišljenje; pobeđuje viši confidence, gubitnik se čuva u novim `alt_value`/
`alt_confidence` poljima (additive, transparentno - nikad se ne krije neslaganje).
Confidence za TrOCR je geometrijska sredina verovatnoća generisanih tokena
(standardna seq2seq mera, ne izmišljen broj).

**Stvarno instaliran i testiran model (ne teorija)** na primer.pdf poljima:
```
srednja_skola_naziv: EasyOCR "XII бсоградска гимназиЈа" (49.3%) 
                   → TrOCR   "XIII београдска гимназита" (93.7%) -> POBEDIO, ugrađeno u odgovor
ime_prezime_studenta: TrOCR "Петарпетровий" (conf≈0.0) -> ispravno odbijeno (EasyOCR ostaje)
mesto_rodjenja/prebivaliste_naselje: TrOCR HALUCINIRA potpuno nepovezan crkvenoslovenski
                   tekst ("И҆ речѐ і҆исꙋ́съ...", "И҆ речѐ а҆враа́мъ...") - model je očigledno
                   treniran na religioznim tekstovima i kad je slika nejasna, "izmišlja"
                   verovatan tekst IZ SVOG TRENING SKUPA umesto da prizna da ne zna.
                   Confidence je ispravno skoro nula (0.000, 0.002) - gating ga ispravno
                   odbacuje, ali ovo je konkretan, izmeren dokaz rizika koji je analiza
                   samo teoretski pretpostavljala.
```

**Izmerena cena**: obrada cele strane sa uključenim flagom porasla je sa ~20s na
**~200 sekundi** (10x) kad se TrOCR pozove za više polja - CPU inferenca transformer
modela je spora. Ovo je razlog zašto je `ENABLE_HANDWRITING_TROCR` ostao `False` po
default-u i zašto je najbolja praksa da se, ako se ikad uključi, koristi ISKLJUČIVO
preko eksplicitnog `"handwriting": true` po polju (ne preko globalnog confidence
praga koji nekontrolisano hvata mnogo polja i čini svaki zahtev sporim).

### ✅ 4.3 — Poštena GPU dijagnostika — GOTOVO

`config.py Config.detect_gpu()` i `ocr_engine.py _detect_hardware()` više ne
proveravaju `onnxruntime` provajdere (nepovezano od onoga što EasyOCR stvarno
koristi) - sad proveravaju `torch.cuda.is_available()` i prisustvo `torch_directml`
paketa, tj. ono što EasyOCR STVARNO koristi. Potvrđeno na ovoj mašini:
`torch==2.10.0+cpu` (CPU-only build), `torch.cuda.is_available()==False`,
`torch_directml` nije instaliran → `/api/health` sad iskreno javlja
`"gpu_available": false, "gpu_backend": "cpu"` umesto ranijeg lažnog "directml".
Stvarna AMD GPU akceleracija (torch-directml integracija) nije pokušana u ovom
krugu - rizično/nepotvrđeno, ostavljeno kao budući korak (vidi 5. Alternative).

### ✅ 4.6 — Uklonjene mrtve OMR fallback grane — GOTOVO

`processors/omr_logic.py`: `_detect_circles_by_contour`, `_detect_by_edge_density`,
`_detect_by_zone_density`, `_prepare_options`, `detect_checkbox` (nekorišćen) i
zastareo `__main__` self-test su obrisani - potvrđeno (skriptom preko stvarnog
`sv20_template.json`) da svih 12 OMR polja u produkciji ima precizne koordinate, pa
se ove grane realno nikad nisu izvršavale. Bez preciznih koordinata sad se vraća
jasna poruka ("definiši koordinate u Template Editoru") umesto nagađanja starim
heuristikama. Fajl smanjen sa 873 na 382 linije. Svi pytest testovi i dalje prolaze.

### ✅ 4.7 — README napomena (JSON vs baza) — GOTOVO

Dodata napomena u `README.md` da je `sv20_template.json` jedini zvaničan izvor
definicije polja - `DB_CONFIG`/MySQL konekcija u `config.py` postoji ali se nigde
ne koristi.

### ✅ 4.9 — CORS + smanjenje log verbosity — GOTOVO

`allow_credentials` promenjen na `False` (CORS wildcard+credentials kombinacija koju
browseri odbijaju - Java klijent ionako ne šalje kolačiće). Per-opcija OMR logovi
(`[OMR SCORES]`, `[OMR MULTI PRECISE]`) i per-polje `[OMR]`/`[SCALE]` logovi na hot
path-u (`/api/ocr/process`) prebačeni sa INFO na DEBUG - `/api/debug/omr-rois`
namerno ostaje na INFO jer je to already-opt-in debug alat.

### ✅ WS7 (novo) — OMR tačnost: ring_score + padding + anchori po strani — GOTOVO, testirano na stvarnom formularu koji si poslao

Posle prethodnog kruga, na tvom stvarnom popunjenom obrascu su 3 od 4 OMR polja na
strani 1 promašila (`vrsta_studija`, `pol`, `bracni_status` - samo `nacin_finansiranja`
je proradio). Izvukao sam stvarne isečke preko `/api/debug/omr-rois` i vizuelno + brojčano
dijagnostikovao **dva odvojena, stvarna uzroka** (ne pretpostavka):

1. **Pravougaonici opcija u Editoru sistematski seku donji deo kruga.** Anotirane slike
   su to jasno pokazale - kod polja koja rade (nacin_finansiranja) box velikodušno
   pokriva ceo krug; kod polja koja ne rade, box se završava TAČNO gde krug još traje.
2. **`fill_ratio`/`edge_score` heuristika ne razlikuje "zaokruženo" od "krivudava
   odštampana cifra".** Cifra "2" ima prirodno više ivica od cifre "1" i bez ikakvog
   kruga - to je dovoljno da izgleda "zaokruženije" od stvarno zaokruženog "1" sa
   tanjom, svetlijom linijom.

**Rešenje** (`processors/omr_logic.py`):
- Automatski **padding** (25% oko svake opcije, testirano i validno u širokom opsegu
  0.15-0.50 - nije nabačeno na jedan primer) koji toleriše nepotpuno poravnate
  pravougaonike.
- Nov **`ring_score`** - detekcija preko HIJERARHIJE kontura (`cv2.RETR_CCOMP`): traži
  konturu koja ima DETE (rupu unutra) - strukturalni potpis "ovo nešto OBAVIJA", za
  razliku od generičke gustine mastila/ivica. `combined_score` koristi `ring_score` kad
  god ga bar jedna opcija u polju ima; inače fallback na stari fill/edge signal.

**Rezultat na tvom skenu** (`primer.pdf`, obe strane, 12 OMR polja):
```
PRE:  4/12 polja tačno detektovano (nekoliko sa lažnim "ne znam")
POSLE: 12/12 polja tačno detektovano, sva sa 91.6%-100% confidence
       (vrsta_studija, pol, bracni_status sad TAČNO prepoznaju isto što i ti vidiš na slici)
```
Ukupno: 26/37 → **34/37** uspešnih polja na celom obrascu. Dodat regresioni test
(`test_ring_score_recovers_circle_cut_by_undersized_option_box`) koji simulira tačno
ovaj scenario (box koji seče krug) da se greška ne ponovi.

**Anchori po strani** (odgovor na "druga strana nema logo/naslov za anchor" i "kockice
po uglovima"): `template.document.anchor_elements` je sad **po stranici**
(`{"1": {...}, "2": {...}}`), ne globalno za ceo dokument - Editor automatski migrira
tvoju već sačuvanu kalibraciju stranice 1. Na stranici 2 (i svakoj sledećoj) Editor
nudi 2 nova sidra sa default pozicijama u **gornjem levom i gornjem desnom uglu** te
strane - prevučeš ih na BILO KOJI deo strane 2 koji ima dovoljno kontrasta (ivica
tabele, broj sekcije, čak i sam ugao papira ako se dovoljno razlikuje od pozadine).
Fizičke "kockice" nisu izvodljive (menjale bi zvaničan obrazac), ali ovo postiže istu
svrhu - ti biraš TAČKU oslonca na svakoj strani posebno, isti alat, samo sad zna da
strane nisu iste. Ako sidro na nekoj strani ne uspe da se nađe (npr. slabo izabran
deo), sistem i dalje gracioznо pada na `manual_fallback` - ništa se ne pokvari.
Potvrđeno na `primer.pdf`: stranica 1 i dalje javlja `alignment_mode_used: "anchor"`
(tvoja postojeća kalibracija radi nepromenjeno), stranica 2 ispravno pada na
`manual_fallback` dok ne dodaš njena sidra - obe strane i dalje daju tačne rezultate.

### Preostalo van dometa ovog kruga (namerno)

- **4.8 (širi test coverage)**: delimično pokriveno usput - 13 pytest testova sada
  postoje (nula pre), uključujući 2 regresiona testa sa stvarnim brojevima sa
  `primer.pdf` skena. Puna pokrivenost `image_processor.py`/`ocr_engine.py` nije
  urađena (nije bila deo eksplicitnog zahteva).
- **Stvarna AMD GPU akceleracija preko torch-directml**: namerno NIJE pokušana -
  zahtevala bi patch-ovanje EasyOCR-ovog internog device placement-a (ne podržava
  custom torch device "od kutije"), rizično i nepotvrđeno u odnosu na koliko bi
  realno ubrzalo servis. Dokumentovano kao budući korak.

### Kontrakt prema relacionom modelu (WS5)

Uporedio JSON odgovor `/api/ocr/process` sa modelom koji si nalepio iznad:

| Relacioni model | JSON polje | Status |
|---|---|---|
| `StavkeObrasca.ocrVrednost` | `ocr_value` | ✅ poklapa se |
| `StavkeObrasca.korigovanaVrednost` | `validated_value` | ✅ poklapa se (kod JMBG-a sad ostaje = ocr_value dok se ne potvrdi is_valid) |
| `StavkeObrasca.nivoPodudarnosti` (DOUBLE) | `confidence` (0–100) | ✅ poklapa se |
| `StavkeObrasca.ocrUspesno` | `is_valid` | ✅ poklapa se |
| `StavkeObrasca.idPolja` / `TipPolja.idPolja` | `field_id` | ✅ poklapa se |
| `TipPolja.nazivPolja` | `field_name` | ✅ poklapa se |
| `TipPolja.pozicijaX/Y/sirina/visina` | `coordinates.{x,y,width,height}` | ✅ poklapa se |
| `TipPolja.stranica` | `field.page` u template-u (ne vraća se eksplicitno u response-u, ali se koristi za filtriranje) | ⚠️ vidi napomenu |
| `TipPolja.redosledObrade` | Implicitno — redosled polja u `template.fields` nizu | ⚠️ vidi napomenu |
| `TipPolja.tipPodatka` (enum TEXT/NUMERIC/DATE/BOOLEAN/ALPHANUMERIC) | `field_type` (uključuje i `OMR_SINGLE`, `OMR_MULTI`, `SIGNATURE`) | ❓ **otvoreno pitanje** |
| `TipPolja.regexValidacija` | Validacija se radi u Python `validators.py`, ne šalje se regex string nazad | ℹ️ informativno |
| `Student.indeks`, `Student.jmbg` | `broj_indeksa`/`jmbg` polja | ✅ poklapa se — ovo su FK/identitetska polja, zato su WS1 i (planirani) WS3 direktno bitni: greška ovde vezuje obrazac za pogrešnog studenta |
| `ZaposleniFakulteta`, `StudijskiProgram`, `TerminDezurstva`, `Zaposleni-Termin` | — | OCR servis ih namerno ne dodiruje — čisto Java/DB entiteti van njegovog domena |

**Otvoreno pitanje za Java/DB stranu**: `tipPodatka` enum u bazi — da li već sadrži
`OMR_SINGLE`/`OMR_MULTI`/`SIGNATURE`, ili se OMR/potpis polja na Java strani modeluju
drugačije (npr. kao `BOOLEAN`/`TEXT` sa dodatnom logikom)? Ovaj Python servis ne može to da
zna bez uvida u stvarnu Java enum definiciju — vredi proveriti pre nego što se GUI osloni na
tačan string `field_type`.

**Nove (opcione) JSON stavke dodate ovom rundom** — additive, ne kvare postojeći Java
parsing ako se ignorišu: `suggested_value` (WS1, samo za JMBG polja sa mogućom ispravkom).