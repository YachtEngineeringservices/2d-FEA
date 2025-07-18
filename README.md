# 2D FEA Analysis Tool

A cross-platform 2D Finite Element Analysis application for structural and torsional analysis.

## Features

- **2D FEA Simple**: Basic structural analysis with interactive GUI
- **2D FEA Torsion Analysis**: Specialized torsional analysis tool with advanced point editing
- **Full DOLFINx FEA Web App**: Professional finite element analysis in your browser with the same solver as desktop
- **Interactive Point Editing**: Edit coordinates, reorder points, and delete points with intuitive controls
- **Interactive Plot Navigation**: Zoom with mouse wheel, pan with right-click drag, click-to-select points
- **Professional Stress Visualization**: True stress field contour plots from DOLFINx mesh
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

### 3. Web Application (Full FEA)

#### Local Development
```bash
# Quick launch with full DOLFINx FEA
./launch_web_app.sh

# Or manually:
# Install web dependencies
pip install -r requirements_web.txt

# Run web app with full FEA support
streamlit run src/web_app.py
```

#### Cloud Deployment (Render.com)
```bash
# Deploy to Render.com with full DOLFINx support
./deploy_render.sh
```

**Available at**: [https://2d-fea-dolfinx.onrender.com](https://2d-fea-dolfinx.onrender.com)

**Web App Features:**
- **Full DOLFINx FEA**: Same professional solver as desktop version
- **GMSH Mesh Generation**: Quality finite element meshes
- **Professional Visualization**: True stress field contour plots
- **Always-on Service**: No cold starts, professional deployment
- **Browser-based**: Access from any web browser worldwide
- **Multi-point Geometry Input**: Paste coordinates directly
- **Cost**: $7/month on Render.com (Starter plan)

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
- **For full FEA capabilities**: DOLFINx, GMSH (install via conda-forge)

### FEA Environment Setup
```bash
# Create conda environment with full FEA support
conda create -n fenics-core -c conda-forge fenics-dolfinx pyvista gmsh h5py meshio
conda activate fenics-core
pip install streamlit plotly
```

## License

Open source - see repository for details.
