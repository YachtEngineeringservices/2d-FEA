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
