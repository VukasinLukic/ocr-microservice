
Šta je urađeno u OCR mikroservisu (C:\Users\Tea\ocr-microservice\)

  sv20-ocr-service/requirements.txt

  Uklonjen onnxruntime-directml — nije bio koristan za EasyOCR (koji koristi PyTorch, ne ONNX), a mogao je blokirati pip
  install na nekim sistemima.

  sv20-ocr-service/api/server.py

  Popravio sam putanju za static direktorijum — koristila se relativna putanja koja ne radi kad se servis pokrenep iz
  drugog foldera. Sada koristi apsolutnu putanju baziranu na __file__.

  Novi fajl: pokreni.bat

  Dvostruki klik = pokretanje servisa. Provjeri Python, provjeri zavisnosti, pokrene iz ispravnog direktorijuma.

  Novi fajl: instaliraj.bat

  Jednokratna instalacija svih Python paketa (easyocr, fastapi, opencv, itd.). Pokreni ovo jednom pre prvog korišćenja.

  ---
  Šta je urađeno u Java projektu

  dokumenti/seed_tippolja.sql (KLJUČNO)

  SQL skripta sa svim 37 TipPolja zapisa čiji nazivi tačno odgovaraju field_name vrijednostima iz OCR template-a
  (sv20_template.json). Bez ove skripte OCR radi ali ne upisuje stavke.

  OcrKlijent.java

  Sada čita validated_value (post-procesirana, čišća verzija) umjesto raw ocr_value.


  Dvostruki klik = pokretanje servisa. Provjeri Python, provjeri zavisnosti, pokrene iz ispravnog direktorijuma.

  Novi fajl: instaliraj.bat

  Jednokratna instalacija svih Python paketa (easyocr, fastapi, opencv, itd.). Pokreni ovo jednom pre prvog korišćenja.

  ---
  Šta je urađeno u Java projektu

  dokumenti/seed_tippolja.sql (KLJUČNO)

  SQL skripta sa svim 37 TipPolja zapisa čiji nazivi tačno odgovaraju field_name vrijednostima iz OCR template-a
  (sv20_template.json). Bez ove skripte OCR radi ali ne upisuje stavke.

  OcrKlijent.java

  Sada čita validated_value (post-procesirana, čišća verzija) umjesto raw ocr_value.

  SV20ObrazacController.java

  Poklapanje polja je sada fleksibilnije (substring matching), a u slučaju da nema poklapanja prikazuje konkretnu poruku
  šta treba uraditi.

  ---
  Redosled pokretanja za demo

  1. Jednom:  instaliraj.bat  (instalira Python pakete ~1GB)

  2. Svaki put:
     a) pokreni.bat            (OCR servis na http://localhost:9001)
     b) Pokreni 0_PsServer     (Java server)
     c) Pokreni PsKlijent      (Java klijent)

  3. U SQL klijentu:
     Pokreni dokumenti/seed_tippolja.sql  (samo jednom, popuni TipPolja)

  Jedini potencijalni problem: Ako si u TipPoljaForma već dodao svoja polja s drugačijim imenima, seed skripta će
  kreirati duplikate — u tom slučaju iskoristi formu da ručno dodaš polja s imenima iz OCR template-a (npr.
  ime_prezime_studenta, jmbg, broj_indeksa...).