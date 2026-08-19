@echo off
chcp 65001 >nul
echo ============================================================
echo  SV-20 OCR Mikroservis - Instalacija zavisnosti
echo ============================================================
echo.

cd /d "%~dp0sv20-ocr-service"

:: Provjeri Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [GRESKA] Python nije pronađen. Instaliraj Python 3.9+
    pause
    exit /b 1
)

echo Python verzija:
python --version
echo.

:: Upgrade pip
echo Azuriranje pip-a...
python -m pip install --upgrade pip

echo.
echo Instaliranje zavisnosti iz requirements.txt...
echo NAPOMENA: easyocr ce preuzeti PyTorch (~1GB). Sacekaj...
echo.

pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [GRESKA] Instalacija nije uspjela. Pogledaj greske iznad.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Instalacija zavrsena!
echo  Pokreni servis sa: pokreni.bat
echo ============================================================
pause
