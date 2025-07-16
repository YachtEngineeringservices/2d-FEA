#!/bin/bash

# 2D FEA Web Application Launcher
# This script sets up and runs the Streamlit web application

echo "🚀 2D FEA Web Application Setup"
echo "================================"
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed"
    echo "Please install pip and try again"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv_web" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv_web
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv_web/bin/activate

# Install requirements
echo "📥 Installing web application dependencies..."
pip install --upgrade pip
pip install -r requirements_web.txt

# Check if Streamlit is working
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit installation failed"
    exit 1
fi

echo "✅ All dependencies installed successfully!"
echo

# Launch the web application
echo "🌐 Starting 2D FEA Web Application..."
echo "📱 The application will open in your default browser"
echo "🔗 URL: http://localhost:8501"
echo
echo "Press Ctrl+C to stop the application"
echo

streamlit run src/web_app.py
