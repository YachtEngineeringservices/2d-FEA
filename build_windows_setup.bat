@echo off
echo ================================================
echo 2D FEA Windows Executable Build Script
echo ================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11+ from python.org
    echo.
    pause
    exit /b 1
)

echo Python found. Checking version...
python --version

REM Create virtual environment
echo.
echo Creating virtual environment...
if exist "venv_windows" (
    echo Removing existing virtual environment...
    rmdir /s /q venv_windows
)

python -m venv venv_windows
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv_windows\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install required packages
echo.
echo Installing required packages...
echo This may take several minutes...

REM Core dependencies
pip install PySide6
pip install matplotlib
pip install numpy
pip install PyInstaller

REM Scientific computing packages
pip install gmsh
pip install meshio
pip install h5py

REM Optional but recommended
pip install scipy

echo.
echo ================================================
echo Dependencies installed successfully!
echo ================================================
echo.

REM Build executable
echo Building Windows executable...
echo This may take 10-15 minutes...
echo.

pyinstaller build_windows.spec

if errorlevel 1 (
    echo.
    echo ERROR: Build failed!
    echo Check the output above for error details.
    echo.
    pause
    exit /b 1
)

echo.
echo ================================================
echo BUILD SUCCESSFUL!
echo ================================================
echo.
echo Executable created: dist\2D_FEA_Windows.exe
echo.
echo To test the application:
echo 1. Navigate to the 'dist' folder
echo 2. Double-click '2D_FEA_Windows.exe'
echo.

REM Optional: Test the executable
set /p test_choice="Would you like to test the executable now? (y/n): "
if /i "%test_choice%"=="y" (
    echo.
    echo Testing executable...
    cd dist
    start 2D_FEA_Windows.exe
    cd ..
)

echo.
echo Build process complete!
echo.
pause
