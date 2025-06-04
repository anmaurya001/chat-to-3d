@echo off
setlocal enabledelayedexpansion

echo Starting installation process...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH. Please install Python first.
    pause
    exit /b 1
)

REM Check if requirements files exist
if not exist requirements-torch.txt (
    echo [ERROR] requirements-torch.txt not found!
    pause
    exit /b 1
)
if not exist requirements-other.txt (
    echo [ERROR] requirements-other.txt not found!
    pause
    exit /b 1
)
if not exist requirements.txt (
    echo [ERROR] requirements.txt not found!
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
) else (
    echo Virtual environment already exists.
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment!
    pause
    exit /b 1
)

REM Update pip and install build tools
echo Updating pip and installing build tools...
python -m pip install --upgrade pip wheel
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip and wheel!
    pause
    exit /b 1
)

python -m pip install setuptools==75.8.2
if errorlevel 1 (
    echo [ERROR] Failed to install setuptools!
    pause
    exit /b 1
)

REM Install torch requirements
echo Installing torch requirements...
pip install -r requirements-torch.txt
if errorlevel 1 (
    echo [ERROR] Failed to install torch requirements!
    pause
    exit /b 1
)

REM Install other requirements
echo Installing other requirements...
pip install -r requirements-other.txt
if errorlevel 1 (
    echo [ERROR] Failed to install other requirements!
    pause
    exit /b 1
)

REM Install requirements.txt
echo Installing requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install POC requirements!
    pause
    exit /b 1
)

REM Check if patchfiles directory and flexicubes.py exist
if not exist .\patchfiles\flexicubes.py (
    echo [ERROR] flexicubes.py not found in patchfiles directory!
    pause
    exit /b 1
)

REM Check if destination directory exists
if not exist .\trellis\trellis\representations\mesh\flexicubes (
    echo [ERROR] Destination directory for flexicubes.py does not exist!
    pause
    exit /b 1
)

REM Copy flexicubes.py to trellis directory
echo Copying flexicubes.py to trellis directory...
copy .\patchfiles\flexicubes.py .\trellis\trellis\representations\mesh\flexicubes\flexicubes.py
if errorlevel 1 (
    echo [ERROR] Failed to copy flexicubes.py!
    pause
    exit /b 1
)

echo Installation completed successfully!
pause 