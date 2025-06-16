@echo off
setlocal enabledelayedexpansion

REM Set Python to unbuffered mode
set PYTHONUNBUFFERED=1

echo Starting installation process...
echo Current directory: %CD%
echo PATH: %PATH%

REM Store the current directory
set "CURRENT_DIR=%CD%"

REM Check if conda is installed
echo Checking conda installation...
where conda >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Conda is not installed or not in PATH. Please install Conda first.
    pause
    exit /b 1
)

echo Conda found, continuing installation...

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
conda env list | findstr "trellis_v2" >nul
if errorlevel 1 (
    echo Creating conda environment 'trellis_v2'...
    conda create -n trellis_v2 python=3.11 -y
    if errorlevel 1 (
        echo [ERROR] Failed to create conda environment!
        pause
        exit /b 1
    )
) else (
    echo Conda environment 'trellis_v2' already exists.
)

REM Update pip and install build tools
echo Updating pip and installing build tools...
call conda run -n trellis_v2 --no-capture-output python -u -m pip install --upgrade pip wheel
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip and wheel!
    pause
    exit /b 1
)

call conda run -n trellis_v2 --no-capture-output python -u -m pip install setuptools==75.8.2
if errorlevel 1 (
    echo [ERROR] Failed to install setuptools!
    pause
    exit /b 1
)

REM Install torch requirements
echo Installing torch requirements...
call conda run -n trellis_v2 --no-capture-output python -u -m pip install -r requirements-torch.txt
if errorlevel 1 (
    echo [ERROR] Failed to install torch requirements!
    pause
    exit /b 1
)

REM Install other requirements
echo Installing other requirements...
call conda run -n trellis_v2 --no-capture-output python -u -m pip install -r requirements-other.txt
if errorlevel 1 (
    echo [ERROR] Failed to install other requirements!
    pause
    exit /b 1
)

REM Install requirements.txt
echo Installing requirements.txt...
call conda run -n trellis_v2 --no-capture-output python -u -m pip install -r requirements.txt
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
cd chat-to-3d-core
call conda run -n trellis_v2 --no-capture-output python -u download_models.py
if errorlevel 1 (
    echo [ERROR] Failed to download models!
    cd "%CURRENT_DIR%"
    pause
    exit /b 1
)
cd "%CURRENT_DIR%"

echo Installation completed successfully!

REM --- Blender Addon Installation ---
echo Starting Blender addon installation...

REM Check if blender folder exists
if not exist blender (
    echo [ERROR] blender folder not found in current directory!
    pause
    exit /b 1
)

REM Check if source files exist
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

REM Set Blender root directory
set "BLENDER_ROOT=%appdata%\Blender Foundation\Blender"
echo Checking Blender installation at: %BLENDER_ROOT%

if not exist "%BLENDER_ROOT%" (
    echo [WARNING] Blender installation not found at: %BLENDER_ROOT%
    echo Please install Blender or manually copy the addon files to your Blender addons directory.
    pause
    exit /b 0
)

REM Function to copy files to a given addons directory (with optional directory creation)
:copy_files
set "DEST_DIR=%~1"
set "CREATE_DIR=%~2"
echo Checking addon directory: !DEST_DIR!
if "!CREATE_DIR!"=="true" (
    if not exist "!DEST_DIR!" (
        echo Creating destination directory: !DEST_DIR!
        mkdir "!DEST_DIR!"
        if errorlevel 1 (
            echo [ERROR] Failed to create destination directory: !DEST_DIR!
            pause
            exit /b 1
        )
    )
) else (
    if not exist "!DEST_DIR!" (
        echo [INFO] Skipping !DEST_DIR! because it does not exist.
        goto :eof
    )
)
echo Copying NV_Trellis_Addon.py to !DEST_DIR!...
copy blender\NV_Trellis_Addon.py "!DEST_DIR!\NV_Trellis_Addon.py"
if errorlevel 1 (
    echo [ERROR] Failed to copy NV_Trellis_Addon.py to !DEST_DIR!
    pause
    exit /b 1
)
echo Copying asset_importer.py to !DEST_DIR!...
copy blender\asset_importer.py "!DEST_DIR!\asset_importer.py"
if errorlevel 1 (
    echo [ERROR] Failed to copy asset_importer.py to !DEST_DIR!
    pause
    exit /b 1
)
echo Successfully copied addon files to !DEST_DIR!
goto :eof

REM Copy files to Blender 4.2 addons directory (create directory if missing)
echo Processing Blender 4.2...
call :copy_files "%BLENDER_ROOT%\4.2\scripts\addons" true

REM Copy files to Blender versions greater than 4.2 (only if directory exists)
echo Checking for Blender versions greater than 4.2...
for /d %%D in ("%BLENDER_ROOT%\*") do (
    set "VERSION=%%~nxD"
    REM Skip non-numeric directories and versions <= 4.2
    echo !VERSION! | findstr /r "^[0-9]*\.[0-9]*$" >nul
    if !errorlevel! equ 0 (
        for /f "tokens=1,2 delims=." %%A in ("!VERSION!") do (
            set /a MAJOR=%%A
            set /a MINOR=%%B
            if !MAJOR! gtr 4 (
                echo Found Blender version !VERSION!, checking for addons directory...
                call :copy_files "%%D\scripts\addons" false
            ) else if !MAJOR! equ 4 (
                if !MINOR! gtr 2 (
                    echo Found Blender version !VERSION!, checking for addons directory...
                    call :copy_files "%%D\scripts\addons" false
                )
            )
        )
    )
)

echo Blender addon installation completed successfully!
echo All tasks completed successfully!
pause