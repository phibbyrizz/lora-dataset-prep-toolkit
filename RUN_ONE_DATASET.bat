@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "DATASET=%~1"
if not defined DATASET (
    echo Select the curated subject folder you want to prepare...
    for /f "usebackq delims=" %%I in (`powershell.exe -NoProfile -STA -Command "Add-Type -AssemblyName System.Windows.Forms; $d=New-Object System.Windows.Forms.FolderBrowserDialog; $d.Description='Select one curated subject dataset folder'; $d.ShowNewFolderButton=$false; if($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK){[Console]::WriteLine($d.SelectedPath)}"`) do set "DATASET=%%I"
)
if not defined DATASET (
    echo.
    echo No dataset folder selected. Nothing was changed.
    pause
    exit /b 0
)
if not exist "%DATASET%\" (
    echo ERROR: Dataset folder does not exist:
    echo   %DATASET%
    pause
    exit /b 1
)
set "PYEXE=%~dp0.prep_venv\Scripts\python.exe"
if not exist "%PYEXE%" (
    echo ERROR: Toolkit environment not found. Run SETUP_WINDOWS.bat first.
    pause
    exit /b 1
)
set "DEFAULT_TRIGGER="
set "TRIGGER_FILE=%TEMP%\lora_prep_trigger_%RANDOM%_%RANDOM%.txt"
"%PYEXE%" "%~dp0trigger_helper.py" "%DATASET%" > "%TRIGGER_FILE%"
if not errorlevel 1 set /p "DEFAULT_TRIGGER=" < "%TRIGGER_FILE%"
del /q "%TRIGGER_FILE%" >nul 2>&1
if not defined DEFAULT_TRIGGER (
    echo ERROR: A trigger could not be derived from the selected folder name.
    echo You will need to enter a trigger manually below.
)

echo.
echo Dataset selected:
echo   %DATASET%
echo.
echo IMPORTANT: The trigger is inserted at the beginning of every caption.
echo Folder names are converted to lowercase triggers; single underscores become spaces.
echo Double underscores create aliases. Example: John_Smith__Johnny
echo becomes: john smith, johnny
echo.
echo Detected trigger: %DEFAULT_TRIGGER%
set "TRIGGER="
set /p "TRIGGER=Press Enter to use it, or type a different trigger: "
if not defined TRIGGER set "TRIGGER=%DEFAULT_TRIGGER%"
if not defined TRIGGGER (
    echo.
    echo ERROR: Trigger cannot be blank. Nothing was processed.
    pause
    exit /b 1
)
echo.
echo Trigger for this run:
echo   %TRIGGER%
echo.
echo Running preflight. On first use, JoyCaption may need a large one-time download...
"%PYEXE%" "%~dp0runtime_preflight.py" --ensure-joycaption
if errorlevel 1 (
    echo.
    echo Preflight failed. No dataset processing was started.
    pause
    exit /b 1
)
"%PYEXE%" "%~dp0build_lora_dataset.py" "%DATASET%" --trigger "%TRIGGER%"
if errorlevel 1 (
    echo.
    echo Pipeline stopped. The failed dataset was not zipped.
    pause
    exit /b 1
)
echo.
echo Dataset preparation completed successfully.
pause
exit /b 0
