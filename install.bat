@echo off
setlocal enabledelayedexpansion

echo Starting installation process...

REM Check if conda is installed
echo Checking for Conda...
where conda >nul 2>&1
set CONDA_ERRORLEVEL=%errorlevel%
echo Conda command returned errorlevel: %CONDA_ERRORLEVEL%
if %CONDA_ERRORLEVEL% NEQ 0 (
    echo [ERROR] Conda is not installed or not in PATH. Please install Conda first.
    pause
    exit /b 1
)
echo Conda check passed, proceeding to file checks...

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

REM Create conda environment if it doesn't exist
conda env list | findstr "trellis" >nul
if errorlevel 1 (
    echo Creating conda environment 'trellis'...
    call conda create -n trellis python=3.11.9 -y
    if errorlevel 1 (
        echo [ERROR] Failed to create conda environment!
        pause
        exit /b 1
    )
) else (
    echo Conda environment 'trellis' already exists.
)

REM Activate conda environment
echo Activating conda environment...
call conda activate trellis
if errorlevel 1 (
    echo [ERROR] Failed to activate conda environment!
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

REM Download required models
echo Downloading required models...
set CURRENT_DIR=%cd%
echo current directory: %CURRENT_DIR%
cd chat-to-3d-core
call conda run -n trellis --no-capture-output python download_models.py
if errorlevel 1 (
    echo [ERROR] Failed to download models!
    cd "%CURRENT_DIR%"
    pause
    exit /b 1
)
echo current directory: %CURRENT_DIR%
cd "%CURRENT_DIR%"

echo Installation completed successfully!



:: Start Blender addon installation
echo Starting Blender addon installation...

:: Check if blender folder exists
if not exist blender (
  echo [ERROR] blender folder not found in current directory!
  pause
  exit /b 1
)

:: Check if source files exist
if not exist blender\NV_Trellis_Addon.py (
  echo [ERROR] NV_Trellis_Addon.py not found in blender folder!
  pause
  exit /b 1
)
if not exist blender\asset_importer.py (
  echo [ERROR] asset_importer.py not found in blender folder!
  pause
  exit /b 1
)

:: Set Blender root directory
set "BLENDER_ROOT=%appdata%\Blender Foundation\Blender"
echo Blender root directory: %BLENDER_ROOT%

:: Install addons for Blender 4.2 and higher versions
echo Checking for Blender versions 4.2 and higher...
for /d %%D in ("%BLENDER_ROOT%\*") do (
  set "VERSION=%%~nxD"
  echo Found directory: !VERSION!
  set "IS_VERSION=0"
  if "!VERSION!"=="4.2" set "IS_VERSION=1"
  if "!VERSION!"=="4.3" set "IS_VERSION=1"
  if "!VERSION!"=="4.4" set "IS_VERSION=1"
  if "!VERSION!"=="4.5" set "IS_VERSION=1"
  if "!VERSION!"=="4.6" set "IS_VERSION=1"
  if "!VERSION!"=="4.7" set "IS_VERSION=1"
  if "!VERSION!"=="4.8" set "IS_VERSION=1"
  if "!VERSION!"=="4.9" set "IS_VERSION=1"
  if "!VERSION!"=="5.0" set "IS_VERSION=1"
  if !IS_VERSION! equ 1 (
    echo Checking version !VERSION!
    set "INSTALL_ADDONS=0"
    for /f "tokens=1,2 delims=." %%A in ("!VERSION!") do (
      set "MAJOR=%%A"
      set "MINOR=%%B"
      if defined MAJOR if defined MINOR (
        set /a MAJOR_NUM=!MAJOR!+0
        set /a MINOR_NUM=!MINOR!+0
        if !MAJOR_NUM! gtr 4 set "INSTALL_ADDONS=1"
        if !MAJOR_NUM! equ 4 if !MINOR_NUM! geq 2 set "INSTALL_ADDONS=1"
      )
    )
    if !INSTALL_ADDONS! equ 1 (
      set "ADDONS_DIR=%%D\scripts\addons"
      echo Processing Blender !VERSION! addons directory: !ADDONS_DIR!
      if not exist "!ADDONS_DIR!" (
        echo Creating addons directory: !ADDONS_DIR!
        mkdir "!ADDONS_DIR!"
        if errorlevel 1 (
          echo [ERROR] Failed to create addons directory: !ADDONS_DIR!
          pause
          exit /b 1
        )
      )
      echo Copying NV_Trellis_Addon.py to !ADDONS_DIR!...
      copy blender\NV_Trellis_Addon.py "!ADDONS_DIR!\NV_Trellis_Addon.py"
      if errorlevel 1 (
        echo [ERROR] Failed to copy NV_Trellis_Addon.py to !ADDONS_DIR!
        pause
        exit /b 1
      )
      echo Copying asset_importer.py to !ADDONS_DIR!...
      copy blender\asset_importer.py "!ADDONS_DIR!\asset_importer.py"
      if errorlevel 1 (
        echo [ERROR] Failed to copy asset_importer.py to !ADDONS_DIR!
        pause
        exit /b 1
      )
      echo Successfully copied addons to !ADDONS_DIR!
    )
  ) else (
    echo Skipping non-version directory: !VERSION!
  )
)

echo Blender addon installation completed successfully!

:: Deactivate conda environment
echo Deactivating conda environment...
call conda deactivate

echo All tasks completed successfully!
pause