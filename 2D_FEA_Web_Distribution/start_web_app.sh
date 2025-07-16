#!/bin/bash
# 2D FEA Web App Launcher for Linux/macOS

echo "🌐 Starting 2D FEA Web Application"
echo "=================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not found"
    echo "Please install Python 3 and try again"
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null && ! command -v pip &> /dev/null; then
    echo "❌ pip is required but not found"
    echo "Please install pip and try again"
    exit 1
fi

# Install dependencies if needed
echo "📦 Checking dependencies..."
python3 -c "import streamlit" 2>/dev/null || {
    echo "📥 Installing Streamlit..."
    pip3 install -r requirements.txt
}

echo "🚀 Starting web application..."
echo "   - Local URL: http://localhost:8501"
echo "   - Press Ctrl+C to stop"
echo ""

# Start the application
python3 -m streamlit run src/web_app.py --server.headless true --server.port 8501
