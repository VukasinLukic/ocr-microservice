@echo off
chcp 65001 >nul
echo ============================================================
echo  SV-20 OCR Mikroservis - Pokretanje
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
    echo [GRESKA] Python nije pronađen u PATH-u.
    echo Instaliraj Python 3.9+ sa https://www.python.org/downloads/
    echo NAPOMENA: ako imas Python instaliran ali i dalje vidis ovu poruku, moguce je
    echo da "python" u PATH-u pokazuje na Microsoft Store alias. Iskljuci ga u:
    echo Settings ^> Apps ^> Advanced app settings ^> App execution aliases.
    pause
    exit /b 1
)

:: Provjeri da li su zavisnosti instalirane
%PY% -c "import fastapi, easyocr, cv2, fitz" >nul 2>&1
if errorlevel 1 (
    echo [UPOZORENJE] Neke zavisnosti nisu instalirane.
    echo Pokreci instaliraj.bat prvo!
    echo.
    echo Da li zelis da instaliras sada? ^(D/N^)
    set /p odgovor=
    if /i "%odgovor%"=="D" (
        %PY% -m pip install -r requirements.txt
    ) else (
        pause
        exit /b 1
    )
)

echo.
echo Pokrecem OCR servis na http://localhost:9001 ...
echo Pritisni Ctrl+C da zaustavis.
echo.
%PY% main.py

pause
