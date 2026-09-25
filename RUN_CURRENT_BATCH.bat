@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"

set "PYEXE=%~dp0.prep_venv\Scripts\python.exe"
if not exist "%PYEXE%" goto :no_python

set "SOURCE_ROOT=%~1"
if not defined SOURCE_ROOT (
    echo Select the parent folder containing your subject folders...
    for /f "usebackq delims=" %%D in (`powershell.exe -NoProfile -STA -Command "Add-Type -AssemblyName System.Windows.Forms; $d=New-Object System.Windows.Forms.FolderBrowserDialog; $d.Description='Select the folder containing subject dataset folders'; $d.ShowNewFolderButton=$false; if($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK){[Console]::WriteLine($d.SelectedPath)}"`) do set "SOURCE_ROOT=%%D"
)
if not defined SOURCE_ROOT goto :cancelled
if not exist "%SOURCE_ROOT%\" goto :no_source

echo Running preflight. On first use, JoyCaption may need a large one-time download...
"%PYEXE%" "%~dp0runtime_preflight.py" --ensure-joycaption
if errorlevel 1 (
    echo Preflight failed. No queue processing was started.
    pause
    exit /b 1
)

echo ============================================================
echo  LoRA Dataset Prep Toolkit v0.1.4
echo  Resumable all-folder queue
echo ============================================================
echo Source root: %SOURCE_ROOT%
echo.
echo Choose a run mode:
echo   1. Five-folder stress test ^(recommended first run^)
echo   2. Process every unfinished folder
echo   3. List the first five folders without processing
echo.
choice /c 123 /n /m "Enter 1, 2, or 3: "
if errorlevel 3 goto :list_only
if errorlevel 2 goto :run_all
if errorlevel 1 goto :run_five

:run_five
"%PYEXE%" "%~dp0run_source_queue.py" "%SOURCE_ROOT%" --limit 5
set "QUEUE_EXIT=%ERRORLEVEL%"
goto :finished
:run_all
"%PYEXE%" "%~dp0run_source_queue.py" "%SOURCE_ROOT%"
set "QUEUE_EXIT=%ERRORLEVEL%"
goto :finished
:list_only
"%PYEXE%" "%~dp0run_source_queue.py" "%SOURCE_ROOT%" --limit 5 --list-only
set "QUEUE_EXIT=%ERRORLEVEL%"
goto :finished
:finished
echo.
if "%QUEUE_EXIT%"=="0" (echo QUEUE COMPLETED WITHOUT PROCESSING FAILURES) else (echo QUEUE FINISHED WITH ONE OR MORE FAILURES)
echo Outputs are under: %SOURCE_ROOT%\_pipeline_output
pause
exit /b %QUEUE_EXIT%

:no_python
echo ERROR: Toolkit environment not found. Run SETUP_WINDOWS.bat first.
pause
exit /b 1
:no_source
echo ERROR: Selected source folder does not exist.
pause
exit /b 1
:cancelled
echo No source folder selected. Nothing was changed.
pause
exit /b 0
