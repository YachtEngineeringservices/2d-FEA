# 2D FEA Torsional Analysis Web Application

A web-based finite element analysis tool for torsional stress analysis of custom cross-sections, built with DOLFINx and Streamlit.

## Features

- **Interactive Geometry Input**: Define custom cross-sections with outer and inner boundaries
- **Advanced FEA Solver**: Uses DOLFINx v0.8.0 for accurate torsional stress calculations
- **Real-time Visualization**: High-quality stress distribution plots with full geometry coverage
- **Adaptive Mesh Refinement**: Automatic mesh optimization for better accuracy
- **Interactive Controls**: Zoom and pan with slider controls for detailed analysis
- **Professional Results**: Complete torsional properties including stiffness, twist angle, and maximum stress

## Recent Improvements

✅ **Fixed Left-side Visualization Issue**: Resolved triangulation problem causing missing stress results on geometry left side  
✅ **Enhanced Clear Geometry**: Button now clears all data including mesh, results, and zoom controls  
✅ **Improved Mesh Generation**: Separated mesh and solve workflows with adaptive refinement  
✅ **Better User Interface**: Slider-based zoom/pan controls and real-time analysis logging

### 2. WSL2/Linux Desktop
```bash
# Quick launch (recommended)
./launch_fea.sh

# Or manually:
# Activate FEA environment
conda activate fenics-core

# Install desktop dependencies (if not already installed)
pip install PySide6 pyvista

# Run applications
python src/main.py                    # 2D FEA Torsion Analysis (GUI)
python src/main.py --torsion         # Same as above
```

**Desktop Features:**
- **Full DOLFINx FEA**: Professional finite element solver
- **Interactive GUI**: PySide6-based desktop application
- **GMSH Integration**: Quality mesh generation
- **3D Visualization**: PyVista-based result visualization
- **Point Editing**: Interactive geometry creation
- **Real-time Analysis**: Immediate FEA results

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
