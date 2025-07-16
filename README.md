# 2D FEA Analysis Tool

A cross-platform 2D Finite Element Analysis application for structural and torsional analysis.

## Features

- **2D FEA Simple**: Basic structural analysis with interactive GUI
- **2D FEA Torsion Analysis**: Specialized torsional analysis tool
- **Cross-platform**: Works on Windows, Linux, and web browsers
- **Automated Windows Builds**: GitHub Actions automatically creates Windows executables

## Usage Options

### 1. Windows Executables (Automated via GitHub Actions)
- Download from GitHub Releases
- No installation required - run directly

### 2. WSL2/Linux
```bash
# Install dependencies
pip install -r requirements.txt

# Run applications
python src/main.py                    # 2D FEA Simple
python src/main.py --torsion         # 2D FEA Torsion Analysis
```

### 3. Web Application
```bash
# Install web dependencies
pip install -r requirements.txt

# Run web app
streamlit run src/web_app.py
```

## Development

### Local Development
1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run applications from `src/` directory

### Windows Builds
Windows executables are automatically built via GitHub Actions when code is pushed to the main branch. Download from the Releases section.

## Requirements

- Python 3.11+
- Dependencies listed in `requirements.txt`
- For full FEA capabilities: DOLFINx (Linux/WSL2 only)

## License

Open source - see repository for details.
