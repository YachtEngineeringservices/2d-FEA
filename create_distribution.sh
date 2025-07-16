#!/bin/bash
# 2D FEA Web App Distribution Package Creator
# Creates a portable package for sharing the web application

echo "🎁 Creating 2D FEA Web App Distribution Package"
echo "================================================"

# Create distribution directory
DIST_DIR="2D_FEA_Web_Distribution"
mkdir -p "$DIST_DIR"

echo "📁 Creating distribution folder: $DIST_DIR"

# Copy essential files
echo "📋 Copying application files..."
cp -r src "$DIST_DIR/"
cp requirements_web.txt "$DIST_DIR/requirements.txt"
cp WEB_APP_GUIDE.md "$DIST_DIR/"

# Create simplified structure (remove unnecessary files)
echo "🧹 Cleaning up for distribution..."
find "$DIST_DIR/src" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find "$DIST_DIR/src" -name "*.pyc" -delete 2>/dev/null || true

# Create launcher scripts for different platforms
echo "🚀 Creating launcher scripts..."

# Linux/macOS launcher
cat > "$DIST_DIR/start_web_app.sh" << 'EOF'
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
EOF

chmod +x "$DIST_DIR/start_web_app.sh"

# Windows launcher
cat > "$DIST_DIR/start_web_app.bat" << 'EOF'
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
EOF

# Create a README for the distribution
cat > "$DIST_DIR/README.md" << 'EOF'
# 2D FEA Web Application - Distribution Package

## 🎯 Quick Start

### Windows Users:
1. Double-click `start_web_app.bat`
2. Wait for installation (first time only)
3. Open browser to http://localhost:8501

### Linux/macOS Users:
1. Open terminal in this folder
2. Run: `./start_web_app.sh`
3. Open browser to http://localhost:8501

## 📋 Requirements

- **Python 3.8+** (download from python.org)
- **Internet connection** (for first-time setup)
- **Modern web browser**

## 📖 Documentation

See `WEB_APP_GUIDE.md` for detailed usage instructions.

## 🔧 Manual Installation

If the launcher doesn't work:

```bash
pip install -r requirements.txt
python -m streamlit run src/web_app.py
```

## 📁 Contents

- `src/web_app.py` - Main application
- `requirements.txt` - Python dependencies
- `start_web_app.*` - Platform launchers
- `WEB_APP_GUIDE.md` - User guide

## 🌟 Features

- Browser-based 2D FEA analysis
- Interactive geometry creation
- Material property presets
- Results visualization
- Cross-platform compatibility

Enjoy! 🎉
EOF

# Create requirements file specifically for web app
cat > "$DIST_DIR/requirements.txt" << 'EOF'
streamlit>=1.28.0
matplotlib>=3.7.0
numpy>=1.24.0
scipy>=1.10.0
plotly>=5.15.0
pandas>=2.0.0
EOF

# Create package info
echo "📄 Creating package information..."
cat > "$DIST_DIR/PACKAGE_INFO.txt" << EOF
2D FEA Web Application Distribution Package
==========================================

Package Created: $(date)
Version: 1.0.0
Platform: Cross-platform (Windows, Linux, macOS)

Contents:
- 2D FEA web application
- Platform launchers
- User documentation
- Dependencies list

Sharing Instructions:
1. Compress this folder into a ZIP file
2. Share with users
3. Recipients extract and run launcher

Technical Support:
- Check WEB_APP_GUIDE.md for troubleshooting
- Ensure Python 3.8+ is installed
- Modern web browser required

Created from: $(pwd)
EOF

# Set permissions for the distribution
chmod -R 755 "$DIST_DIR"

echo ""
echo "✅ Distribution package created successfully!"
echo ""
echo "📁 Package location: $DIST_DIR"
echo "📏 Package contents:"
ls -la "$DIST_DIR"
echo ""
echo "🎁 To share this application:"
echo "   1. Compress '$DIST_DIR' into a ZIP file"
echo "   2. Share the ZIP file with users"
echo "   3. Recipients extract and run the launcher script"
echo ""
echo "🧪 Test the package:"
echo "   cd '$DIST_DIR' && ./start_web_app.sh"
echo ""

# Create a ZIP package if zip is available
if command -v zip &> /dev/null; then
    echo "📦 Creating ZIP package..."
    zip -r "${DIST_DIR}.zip" "$DIST_DIR" > /dev/null
    echo "✅ ZIP package created: ${DIST_DIR}.zip"
    echo "📏 ZIP size: $(du -h "${DIST_DIR}.zip" | cut -f1)"
else
    echo "💡 Tip: Install 'zip' to automatically create a ZIP package"
fi

echo ""
echo "🎉 Distribution package ready for sharing!"
