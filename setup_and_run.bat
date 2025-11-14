@echo off
setlocal enabledelayedexpansion

rem -----------------------------------------------------------------
rem Configuration
rem -----------------------------------------------------------------
for %%I in ("%~dp0.") do set "PROJECT_ROOT=%%~fI"
echo [DEBUG] Project root resolved to %PROJECT_ROOT%
set "PYTHONPATH=%PROJECT_ROOT%"
set "REQUIREMENTS_FILE=%PROJECT_ROOT%\requirements.txt"
set "PIPELINE_ENTRY=%PROJECT_ROOT%\run_pipeline.py"
set "BUNDLED_ZIP=%PROJECT_ROOT%\runtime\python-3.13.9-embed-amd64.zip"
set "BUNDLED_DIR=%PROJECT_ROOT%\runtime\python-3.13.9-embed-amd64"
set "BUNDLED_PYTHON=%BUNDLED_DIR%\python.exe"
set "PTH_FILE=%BUNDLED_DIR%\python313._pth"
set "SITE_PACKAGES=%BUNDLED_DIR%\Lib\site-packages"
set "GET_PIP=%PROJECT_ROOT%\runtime\get-pip.py"

echo [INFO] Starting environment bootstrap for fryum pipeline.

rem -----------------------------------------------------------------
rem Prepare bundled Python interpreter (mandatory)
rem -----------------------------------------------------------------
echo [DEBUG] Looking for embedded runtime at %BUNDLED_ZIP%
if exist "%BUNDLED_ZIP%" (
    if not exist "%BUNDLED_PYTHON%" (
        echo [INFO] Extracting embedded Python from %BUNDLED_ZIP%.
        powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -LiteralPath '%BUNDLED_ZIP%' -DestinationPath '%BUNDLED_DIR%' -Force"
        if errorlevel 1 (
            echo [ERROR] Failed to extract embedded Python archive.
            exit /b 1
        )
    ) else (
        echo [INFO] Bundled Python already extracted in %BUNDLED_DIR%.
    )
) 
echo [DEBUG] Extraction block completed
if not exist "%BUNDLED_PYTHON%" (
    echo [ERROR] Missing embedded runtime.
    echo         Place python-3.13.9-embed-amd64.zip under %PROJECT_ROOT%\runtime\
    echo         ^(the script extracts it automatically^).
    exit /b 1
)

if exist "%PTH_FILE%" (
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "$pth='%PTH_FILE%';" ^
        "if (Test-Path $pth) {" ^
        "  $lines = Get-Content $pth;" ^
        "  $changed = $false;" ^
        "  $lines = $lines | ForEach-Object {" ^
        "    if ($_ -match '^^\s*#import site') { $changed = $true; 'import site' }" ^
        "    else { $_ }" ^
        "  };" ^
        "  if (-not ($lines -match '^^\s*import site')) {" ^
        "    $lines += 'import site';" ^
        "    $changed = $true;" ^
        "  }" ^
        "  if ($changed) { $lines | Set-Content $pth }" ^
        "}"
)

set "PYTHON_EXE=%BUNDLED_PYTHON%"
echo [INFO] Using embedded Python interpreter: %PYTHON_EXE%

if not exist "%SITE_PACKAGES%" (
    mkdir "%SITE_PACKAGES%"
)

rem -----------------------------------------------------------------
rem Ensure pip is available inside embedded interpreter
rem -----------------------------------------------------------------
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$pth='%PTH_FILE%';" ^
    "if (Test-Path $pth) {" ^
    "  $lines = Get-Content $pth;" ^
    "  if (-not ($lines -match 'Lib\\site-packages')) {" ^
    "    Add-Content -Path $pth -Value 'Lib\site-packages'" ^
    "  }" ^
    "  if (-not ($lines -match '\.\.\\\.\.')) {" ^
    "    Add-Content -Path $pth -Value '..\..'" ^
    "  }" ^
    "}"

"%PYTHON_EXE%" -m pip --version >nul 2>&1
if errorlevel 1 (
    if not exist "%GET_PIP%" (
        echo [INFO] Downloading get-pip.py helper.
        powershell -NoProfile -ExecutionPolicy Bypass -Command ^
            "Invoke-WebRequest -Uri https://bootstrap.pypa.io/get-pip.py -OutFile '%GET_PIP%'"
        if errorlevel 1 (
            echo [ERROR] Failed to download get-pip.py.
            exit /b 1
        )
    )
    echo [INFO] Installing pip into embedded interpreter.
    "%PYTHON_EXE%" "%GET_PIP%"
    if errorlevel 1 (
        echo [ERROR] get-pip installation failed.
        exit /b 1
    )
)

echo [INFO] Upgrading pip.
"%PYTHON_EXE%" -m pip install --upgrade pip
if errorlevel 1 (
    echo [WARN] pip upgrade reported errors.
)

if exist "%REQUIREMENTS_FILE%" (
    echo [INFO] Installing Python requirements.
    "%PYTHON_EXE%" -m pip install -r "%REQUIREMENTS_FILE%"
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed.
        exit /b 1
    )
) else (
    echo [WARN] Requirements file not found at %REQUIREMENTS_FILE%.
)

rem -----------------------------------------------------------------
rem Execute the pipeline
rem -----------------------------------------------------------------
if exist "%PIPELINE_ENTRY%" (
    echo [INFO] Running fryum pipeline via %PIPELINE_ENTRY%.
    "%PYTHON_EXE%" "%PIPELINE_ENTRY%"
    if errorlevel 1 (
        echo [ERROR] Pipeline execution failed.
        exit /b 1
    )
) else (
    echo [ERROR] Could not locate %PIPELINE_ENTRY%.
    exit /b 1
)

echo [INFO] All steps completed successfully.
endlocal

