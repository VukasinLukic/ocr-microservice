@echo off
chcp 65001 >nul
echo ============================================================
echo  SV-20 OCR Mikroservis - Instalacija zavisnosti
echo ============================================================
echo.

cd /d "%~dp0sv20-ocr-service"

:: Nadji Python (python/python3 mogu biti Windows Store aliasi koji ne rade)
set "PY="
py --version >nul 2>&1 && set "PY=py"
if not defined PY (
    python --version >nul 2>&1 && set "PY=python"
)
if not defined PY (
    python3 --version >nul 2>&1 && set "PY=python3"
)
if not defined PY (
    echo [GRESKA] Python nije pronađen. Instaliraj Python 3.9+ sa https://www.python.org/downloads/
    echo NAPOMENA: ako imas Python instaliran ali i dalje vidis ovu poruku, moguce je
    echo da "python" u PATH-u pokazuje na Microsoft Store alias. Iskljuci ga u:
    echo Settings ^> Apps ^> Advanced app settings ^> App execution aliases.
    pause
    exit /b 1
)

echo Python verzija:
%PY% --version
echo.

:: Upgrade pip
echo Azuriranje pip-a...
%PY% -m pip install --upgrade pip

echo.
echo Instaliranje zavisnosti iz requirements.txt...
echo NAPOMENA: easyocr ce preuzeti PyTorch (~1GB). Sacekaj...
echo.

%PY% -m pip install -r requirements.txt

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
