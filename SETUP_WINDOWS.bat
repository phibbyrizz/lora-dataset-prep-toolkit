@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHONUTF8=1"

echo ============================================================
echo  LoRA Dataset Prep Toolkit v0.1.4 - Windows setup
echo ============================================================
echo.
where python >nul 2>nul
if errorlevel 1 (
  echo ERROR: Python was not found.
  echo Install Python 3.10-3.12, then run this file again.
  pause
  exit /b 1
)

python -c "import sys; sys.exit(0 if sys.version_info.major == 3 and sys.version_info.minor in (10,11,12) else 1)" >nul 2>nul
if errorlevel 1 (
  echo ERROR: This toolkit currently expects Python 3.10-3.12.
  python --version
  pause
  exit /b 1
)

where nvidia-smi >nul 2>nul
if errorlevel 1 (
  echo ERROR: No NVIDIA driver / nvidia-smi was detected.
  echo This release requires an NVIDIA CUDA-capable GPU for JoyCaption.
  echo Install/update the NVIDIA driver, then run setup again.
  pause
  exit /b 1
)

echo NVIDIA GPU detected:
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

echo.
if not exist ".prep_venv\Scripts\python.exe" (
  echo Creating .prep_venv...
  python -m venv .prep_venv
  if errorlevel 1 goto :fail
)

".prep_venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :fail

echo.
echo Checking CUDA-enabled PyTorch...
".prep_venv\Scripts\python.exe" -c "import torch,sys; print('PyTorch:',torch.__version__); print('CUDA available:',torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>nul
if errorlevel 1 (
  echo CUDA PyTorch is not ready. Installing the official CUDA 12.8 build...
  ".prep_venv\Scripts\python.exe" -m pip uninstall -y torch torchvision torchaudio >nul 2>nul
  ".prep_venv\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  if errorlevel 1 goto :torchfail
)

echo.
echo Verifying GPU acceleration...
".prep_venv\Scripts\python.exe" -c "import torch,sys; print('PyTorch:',torch.__version__); print('PyTorch CUDA:',torch.version.cuda); print('CUDA available:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); sys.exit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 goto :torchfail

echo.
echo Installing toolkit dependencies...
".prep_venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :fail

echo.
echo Running final import check...
".prep_venv\Scripts\python.exe" -c "import torch,cv2,numpy,PIL,transformers,accelerate,bitsandbytes; print('All required packages imported successfully.'); print('GPU ready:',torch.cuda.get_device_name(0))"
if errorlevel 1 goto :fail

echo.
echo Checking required runtime assets...
".prep_venv\Scripts\python.exe" runtime_preflight.py
if errorlevel 1 goto :fail

echo.
echo Recording installed package versions for troubleshooting...
".prep_venv\Scripts\python.exe" -m pip freeze > INSTALLED_VERSIONS.txt

echo.
echo ============================================================
echo  Setup complete - GPU acceleration and runtime assets verified.
echo ============================================================
echo Use RUN_ONE_DATASET.bat for one folder or RUN_CURRENT_BATCH.bat for a queue.
pause
exit /b 0

:torchfail
echo.
echo ERROR: CUDA-enabled PyTorch could not be installed or could not see your NVIDIA GPU.
echo Update your NVIDIA driver and run SETUP_WINDOWS.bat again.
echo For current official PyTorch installation options, see:
echo https://pytorch.org/get-started/locally/
pause
exit /b 3

:fail
echo.
echo Setup failed. Review the error above.
pause
exit /b 1
