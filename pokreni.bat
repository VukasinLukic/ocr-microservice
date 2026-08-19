@echo off
chcp 65001 >nul
echo ============================================================
echo  SV-20 OCR Mikroservis - Pokretanje
echo ============================================================
echo.

cd /d "%~dp0sv20-ocr-service"

:: Provjeri da li Python postoji
python --version >nul 2>&1
if errorlevel 1 (
    echo [GRESKA] Python nije pronađen u PATH-u.
    echo Instaliraj Python 3.9+ sa https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Provjeri da li su zavisnosti instalirane
python -c "import fastapi, easyocr, cv2, fitz" >nul 2>&1
if errorlevel 1 (
    echo [UPOZORENJE] Neke zavisnosti nisu instalirane.
    echo Pokreci instaliraj.bat prvo!
    echo.
    echo Da li zelis da instaliras sada? (D/N)
    set /p odgovor=
    if /i "%odgovor%"=="D" (
        pip install -r requirements.txt
    ) else (
        pause
        exit /b 1
    )
)

echo.
echo Pokrecem OCR servis na http://localhost:9001 ...
echo Pritisni Ctrl+C da zaustavis.
echo.
python main.py

pause
