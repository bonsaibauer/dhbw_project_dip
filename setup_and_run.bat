@echo off
setlocal enabledelayedexpansion

rem -----------------------------------------------------------------
rem Configuration
rem -----------------------------------------------------------------
for %%I in ("%~dp0.") do set "PROJECT_ROOT=%%~fI"
echo [DEBUG] Project root resolved to %PROJECT_ROOT%
set "PYTHONPATH=%PROJECT_ROOT%"
set "REQUIREMENTS_FILE=%PROJECT_ROOT%\requirements.txt"
set "UI_ENTRY=%PROJECT_ROOT%\app\ui\main.py"
set "UI_MODULE=app.ui.main"
set "INSTALLER=%PROJECT_ROOT%\runtime\python-3.11.9-amd64.exe"
set "PYTHON_HOME=%PROJECT_ROOT%\runtime\python311"
set "BUNDLED_PYTHON=%PYTHON_HOME%\python.exe"
set "SCRIPTS_DIR=%PYTHON_HOME%\Scripts"

echo [INFO] Starting environment bootstrap for fryum pipeline.

rem -----------------------------------------------------------------
rem Install bundled Python runtime if necessary
rem -----------------------------------------------------------------
if not exist "%BUNDLED_PYTHON%" (
    if not exist "%INSTALLER%" (
        echo [ERROR] Missing installer at %INSTALLER%.
        exit /b 1
    )
    echo [INFO] Installing Python from %INSTALLER%.
    "%INSTALLER%" /quiet InstallAllUsers=0 Include_launcher=0 Include_pip=1 Include_tcltk=1 ^
        Include_test=0 PrependPath=0 TargetDir="%PYTHON_HOME%"
    if errorlevel 1 (
        echo [ERROR] Python installer failed.
        exit /b 1
    )
) else (
    echo [INFO] Using existing Python installation in %PYTHON_HOME%.
)

set "PYTHON_EXE=%BUNDLED_PYTHON%"

if not exist "%SCRIPTS_DIR%" (
    mkdir "%SCRIPTS_DIR%"
)

rem Ensure pip exists
"%PYTHON_EXE%" -m pip --version >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing pip via ensurepip.
    "%PYTHON_EXE%" -m ensurepip --upgrade
)

rem -----------------------------------------------------------------
rem Prepare dependencies
rem -----------------------------------------------------------------
echo [INFO] Upgrading pip.
"%PYTHON_EXE%" -m pip install --upgrade pip
if errorlevel 1 (
    echo [WARN] pip upgrade reported warnings.
)

if exist "%REQUIREMENTS_FILE%" (
    echo [INFO] Installing Python requirements.
    "%PYTHON_EXE%" -m pip install -r "%REQUIREMENTS_FILE%"
    if errorlevel 1 (
        echo [INFO] Pip not found, installing ensurepip bundle.
        "%PYTHON_EXE%" -m ensurepip --upgrade
        if errorlevel 1 (
            echo [ERROR] ensurepip failed.
            exit /b 1
        )
        "%PYTHON_EXE%" -m pip install --upgrade pip
        "%PYTHON_EXE%" -m pip install -r "%REQUIREMENTS_FILE%"
        if errorlevel 1 (
            echo [ERROR] Dependency installation failed.
            exit /b 1
        )
    )
) else (
    echo [WARN] Requirements file not found at %REQUIREMENTS_FILE%.
)

rem -----------------------------------------------------------------
rem Launch the Tkinter dashboard (pipeline runs in the background)
rem -----------------------------------------------------------------
if exist "%UI_ENTRY%" (
    echo [INFO] Launching Fryum dashboard via module %UI_MODULE%.
    "%PYTHON_EXE%" -m %UI_MODULE%
    if errorlevel 1 (
        echo [ERROR] Dashboard execution failed.
        exit /b 1
    )
) else (
    echo [ERROR] Could not locate %UI_ENTRY%.
    exit /b 1
)

echo [INFO] All steps completed successfully.
endlocal
