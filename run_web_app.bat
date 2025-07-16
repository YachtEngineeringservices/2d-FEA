@echo off
echo 🚀 2D FEA Web Application Setup
echo ================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo ✅ Python found
python --version

REM Create virtual environment if it doesn't exist
if not exist "venv_web" (
    echo 📦 Creating virtual environment...
    python -m venv venv_web
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv_web\Scripts\activate.bat

REM Install requirements
echo 📥 Installing web application dependencies...
python -m pip install --upgrade pip
pip install -r requirements_web.txt

echo ✅ All dependencies installed successfully!
echo.

REM Launch the web application
echo 🌐 Starting 2D FEA Web Application...
echo 📱 The application will open in your default browser
echo 🔗 URL: http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo.

streamlit run src/web_app.py

pause
