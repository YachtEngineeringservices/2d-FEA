@echo off
REM 2D FEA Web App Launcher for Windows

echo 🌐 Starting 2D FEA Web Application
echo ==================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is required but not found
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

REM Check if pip is available
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ pip is required but not found
    echo Please ensure pip is installed with Python
    pause
    exit /b 1
)

REM Install dependencies if needed
echo 📦 Checking dependencies...
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo 📥 Installing Streamlit...
    pip install -r requirements.txt
)

echo 🚀 Starting web application...
echo    - Local URL: http://localhost:8501
echo    - Press Ctrl+C to stop
echo.

REM Start the application
python -m streamlit run src/web_app.py --server.headless true --server.port 8501

pause
