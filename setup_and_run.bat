@echo off
setlocal enabledelayedexpansion

rem -----------------------------------------------------------------
rem Configuration
rem -----------------------------------------------------------------
for %%I in ("%~dp0.") do set "PROJECT_ROOT=%%~fI"

rem Allow skipping the pause via "--no-pause" or "/nopause"
set "PAUSE_ON_EXIT=1"
set "PY_INSTALL_LOG=%PROJECT_ROOT%\python_install.log"
if /I "%~1"=="--no-pause" (
    set "PAUSE_ON_EXIT=0"
    shift
) else if /I "%~1"=="/nopause" (
    set "PAUSE_ON_EXIT=0"
    shift
)

set "EXIT_CODE=0"
set "PYTHON_SOURCE=bundled"
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
call :ENSURE_PYTHON_RUNTIME
if errorlevel 1 goto :FAIL

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
            goto :FAIL
        )
        "%PYTHON_EXE%" -m pip install --upgrade pip
        "%PYTHON_EXE%" -m pip install -r "%REQUIREMENTS_FILE%"
        if errorlevel 1 (
            echo [ERROR] Dependency installation failed.
            goto :FAIL
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
        goto :FAIL
    )
) else (
    echo [ERROR] Could not locate %UI_ENTRY%.
    goto :FAIL
)

echo [INFO] All steps completed successfully.
goto :SUCCESS

:FAIL
set "EXIT_CODE=1"
goto :FINALIZE

:SUCCESS
set "EXIT_CODE=0"
goto :FINALIZE

:ENSURE_PYTHON_RUNTIME
call :TRY_EXISTING_PYTHON
if not errorlevel 1 exit /b 0

if exist "%BUNDLED_PYTHON%" (
    echo [INFO] Using existing Python installation in %PYTHON_HOME%.
    set "PYTHON_EXE=%BUNDLED_PYTHON%"
    exit /b 0
)

if not exist "%INSTALLER%" (
    echo [ERROR] Missing installer at %INSTALLER%.
    exit /b 1
)

echo [WARN] Bundled Python interpreter not found; attempting a repair install.
echo [INFO] Cleaning up previous Python registrations (if any).
"%INSTALLER%" /uninstall /quiet InstallAllUsers=0 >nul 2>&1

if exist "%PY_INSTALL_LOG%" del "%PY_INSTALL_LOG%" >nul 2>&1
echo [INFO] Installing Python from %INSTALLER%.
"%INSTALLER%" /quiet InstallAllUsers=0 Include_launcher=0 Include_pip=1 Include_tcltk=1 ^
    Include_test=0 PrependPath=0 TargetDir="%PYTHON_HOME%" /log "%PY_INSTALL_LOG%"
if errorlevel 1 (
    echo [ERROR] Python installer failed; see %PY_INSTALL_LOG% for details.
    exit /b 1
)

if not exist "%BUNDLED_PYTHON%" (
    echo [ERROR] Python executable missing even after installation attempt.
    echo [ERROR] Refer to %PY_INSTALL_LOG% for more diagnostics.
    exit /b 1
)

echo [INFO] Bundled Python installed successfully.
set "PYTHON_EXE=%BUNDLED_PYTHON%"
exit /b 0

:TRY_EXISTING_PYTHON
set "CANDIDATE_COUNT=0"
set "LAST_VALID_PYTHON="
set "LAST_VALID_VERSION="
set "LAST_VERSION_IS_NEWER=0"

rem Prefer the py launcher if it can provide a 3.11 interpreter
for /f "usebackq delims=" %%P in (`py -3.11 -c "import sys;print(sys.executable)" 2^>nul`) do (
    call :ADD_VALID_PYTHON "%%P"
)

for /f "usebackq delims=" %%C in (`where python 2^>nul`) do (
    call :ADD_VALID_PYTHON "%%C"
)

if %CANDIDATE_COUNT% EQU 0 (
    exit /b 1
)

if %CANDIDATE_COUNT% EQU 1 (
    set "PYTHON_EXE=!PY_CANDIDATE_1!"
    set "PYTHON_SOURCE=system"
    set "PY_SELECTED_VERSION=!PY_VER_1!"
    set "PY_SELECTED_NEWER=!PY_IS_NEWER_1!"
    echo [INFO] Found Python !PY_SELECTED_VERSION! at !PYTHON_EXE!.
    if "!PY_SELECTED_NEWER!"=="1" (
        echo [WARN] Diese Pipeline ist nur mit Python 3.11.9 getestet worden. Neuere Versionen koennen Probleme verursachen.
    )
    exit /b 0
)

echo [INFO] Multiple Python 3.11+ interpreters detected:
for /L %%I in (1,1,%CANDIDATE_COUNT%) do (
    call set "CAND_PATH=%%PY_CANDIDATE_%%I%%"
    call set "CAND_VER=%%PY_VER_%%I%%"
    call set "CAND_NEWER=%%PY_IS_NEWER_%%I%%"
    if "!CAND_NEWER!"=="1" (
        echo     %%I) !CAND_PATH! (Python !CAND_VER! - moeglicherweise inkompatibel ^> 3.11.9)
    ) else (
        echo     %%I) !CAND_PATH! (Python !CAND_VER!)
    )
)
call :WARN_IF_NEWER_PRESENT
call :PROMPT_PYTHON_SELECTION
exit /b %ERRORLEVEL%

:ADD_VALID_PYTHON
set "RAW_PATH=%~1"
if not defined RAW_PATH exit /b 1
call :PROBE_PYTHON "%RAW_PATH%"
if errorlevel 1 exit /b 1
call :REGISTER_PYTHON_CANDIDATE "%LAST_VALID_PYTHON%" "%LAST_VALID_VERSION%" "%LAST_VERSION_IS_NEWER%"
exit /b 0

:REGISTER_PYTHON_CANDIDATE
set "NEW_PATH=%~1"
set "NEW_VERSION=%~2"
set "NEW_IS_NEWER=%~3"
if not defined NEW_PATH exit /b 1
if not defined NEW_VERSION set "NEW_VERSION=unknown"
if not defined NEW_IS_NEWER set "NEW_IS_NEWER=0"
if %CANDIDATE_COUNT% GTR 0 (
    for /L %%I in (1,1,%CANDIDATE_COUNT%) do (
        if /I "!PY_CANDIDATE_%%I!"=="%NEW_PATH%" (
            exit /b 0
        )
    )
)
set /a CANDIDATE_COUNT+=1
set "PY_CANDIDATE_%CANDIDATE_COUNT%=%NEW_PATH%"
set "PY_VER_%CANDIDATE_COUNT%=%NEW_VERSION%"
set "PY_IS_NEWER_%CANDIDATE_COUNT%=%NEW_IS_NEWER%"
exit /b 0

:PROBE_PYTHON
set "CANDIDATE=%~1"
if not defined CANDIDATE exit /b 1
set "LAST_VALID_PYTHON="
set "LAST_VALID_VERSION="
set "LAST_VERSION_IS_NEWER=0"
for /f "usebackq tokens=1,2,3 delims=|" %%A in (`"%CANDIDATE%" -c "import os,platform,sys;sys.exit(1) if sys.version_info < (3,11) else print(f'{os.path.abspath(sys.executable)}|{platform.python_version()}|{int(sys.version_info[:3]>(3,11,9))}')" 2^>nul`) do (
    set "LAST_VALID_PYTHON=%%~A"
    set "LAST_VALID_VERSION=%%~B"
    set "LAST_VERSION_IS_NEWER=%%~C"
)
if not defined LAST_VALID_PYTHON exit /b 1
exit /b 0

:WARN_IF_NEWER_PRESENT
set "HAS_NEWER=0"
for /L %%I in (1,1,%CANDIDATE_COUNT%) do (
    call set "FLAG_NEWER=%%PY_IS_NEWER_%%I%%"
    if "!FLAG_NEWER!"=="1" set "HAS_NEWER=1"
)
if "%HAS_NEWER%"=="1" (
    echo [WARN] Diese Pipeline wurde nur mit Python 3.11.9 verifiziert. Neuere Versionen koennen Fehlverhalten zeigen.
)
exit /b 0

:PROMPT_PYTHON_SELECTION
set "USER_SELECTION="
:PROMPT_USER_CHOICE
set /p "USER_SELECTION=Select interpreter [1-%CANDIDATE_COUNT%] (default 1): "
if "%USER_SELECTION%"=="" set "USER_SELECTION=1"
echo.%USER_SELECTION%| findstr /R "^[0-9][0-9]*$" >nul 2>&1
if errorlevel 1 (
    echo [WARN] Please enter a numeric choice.
    goto :PROMPT_USER_CHOICE
)
if %USER_SELECTION% LSS 1 (
    echo [WARN] Choice must be between 1 and %CANDIDATE_COUNT%.
    goto :PROMPT_USER_CHOICE
)
if %USER_SELECTION% GTR %CANDIDATE_COUNT% (
    echo [WARN] Choice must be between 1 and %CANDIDATE_COUNT%.
    goto :PROMPT_USER_CHOICE
)
call set "PYTHON_EXE=%%PY_CANDIDATE_%USER_SELECTION%%"
call set "PY_SELECTED_VERSION=%%PY_VER_%USER_SELECTION%%"
call set "PY_SELECTED_NEWER=%%PY_IS_NEWER_%USER_SELECTION%%"
set "PYTHON_SOURCE=system"
echo [INFO] Using Python !PY_SELECTED_VERSION! at !PYTHON_EXE!.
if "!PY_SELECTED_NEWER!"=="1" (
    echo [WARN] Diese Pipeline ist nur mit Python 3.11.9 getestet worden. Neuere Versionen koennen Probleme verursachen.
)
exit /b 0

:FINALIZE
if "%PAUSE_ON_EXIT%"=="1" (
    echo.
    echo [INFO] Script completed. Press any key to close this window.
    pause >nul
)
endlocal & exit /b %EXIT_CODE%
