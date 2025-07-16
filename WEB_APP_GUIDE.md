# 2D FEA Web Application - User Guide

## 🌐 Web-Based 2D Finite Element Analysis Tool

This browser-based application allows you to perform 2D finite element analysis without any installation requirements beyond Python and a web browser.

## 🚀 Quick Start

### Option 1: Direct Access (If Already Running)
The web app is currently running at:
- **Local URL**: http://localhost:8501
- **Network URL**: http://172.31.35.7:8501
- **External URL**: http://75.83.145.28:8501

### Option 2: Start the Application

#### On Linux/WSL2:
```bash
cd "/home/adminlinux/2d FEA"
conda activate fenics-core
streamlit run src/web_app.py --server.headless true --server.port 8501
```

#### On Windows (with conda/miniconda):
```batch
cd "C:\path\to\2d FEA"
conda activate fenics-core
streamlit run src/web_app.py --server.headless true --server.port 8501
```

## 📋 Features

### ✅ What Works in the Web App:
- **Interactive Geometry Input**: Click to create custom shapes
- **Predefined Shapes**: Rectangle, Circle, L-Shape, T-Shape
- **Material Properties**: Steel, Aluminum, Concrete presets
- **Simplified FEA Analysis**: Fast analytical solutions
- **Results Visualization**: Stress distribution plots
- **Data Export**: JSON format for results
- **Cross-Platform**: Works on any device with a web browser

### ⚠️ Limitations (vs Full Linux Version):
- Uses simplified analytical methods instead of DOLFINx
- No advanced mesh visualization
- Limited to basic torsion analysis
- No complex boundary conditions

## 🎮 How to Use

### 1. Create Geometry
- **Option A**: Use predefined shapes from the dropdown
- **Option B**: Click "Custom Shape" and click on the plot to add points
- Click "Clear Points" to start over

### 2. Set Material Properties
- Choose from presets (Steel, Aluminum, Concrete)
- Or enter custom values:
  - Young's Modulus (E) in Pa
  - Poisson's Ratio (ν)

### 3. Analysis Parameters
- **Twist per Length**: Angular twist (rad/m)
- **Applied Torque**: External torque (N⋅m)

### 4. Run Analysis
- Click "Run FEA Analysis"
- View results:
  - Maximum stress
  - Total torque
  - Stress distribution plot

### 5. Export Results
- Click "Export Results as JSON"
- Download includes geometry, materials, and results

## 🔧 Troubleshooting

### Common Issues:

1. **App won't start**:
   ```bash
   # Check if conda environment is active
   conda list streamlit
   
   # Reinstall if needed
   pip install streamlit plotly
   ```

2. **Browser can't connect**:
   - Try http://localhost:8501
   - Check firewall settings
   - Ensure port 8501 is available

3. **Analysis fails**:
   - Ensure shape has at least 3 points
   - Check material properties are positive
   - Try predefined shapes first

### Error Messages:
- "Invalid geometry": Shape needs more points or is self-intersecting
- "Analysis failed": Check material parameters
- "Export failed": Browser blocked download - allow popups

## 📊 Technical Details

### Analysis Method:
- **Torsion Analysis**: Uses Prandtl stress function approximation
- **Mesh Generation**: Delaunay triangulation with scipy
- **Stress Calculation**: Analytical formulas for simple geometries
- **Visualization**: Matplotlib with interpolated stress fields

### Performance:
- **Small shapes** (< 100 elements): Instant results
- **Medium shapes** (< 1000 elements): < 5 seconds
- **Large shapes** (> 1000 elements): May take longer

## 🎯 Examples

### Example 1: Steel Rectangular Beam
1. Select "Rectangle" from predefined shapes
2. Choose "Steel" material
3. Set twist per length: 0.1 rad/m
4. Run analysis
5. Expected: ~9.6 GPa max stress

### Example 2: Custom L-Shape
1. Select "Custom Shape"
2. Click points: (0,0), (2,0), (2,0.5), (0.5,0.5), (0.5,2), (0,2)
3. Choose "Aluminum" material
4. Run analysis

## 📱 Device Compatibility

### ✅ Tested On:
- Chrome, Firefox, Safari, Edge
- Windows 10/11
- macOS
- Linux (Ubuntu, WSL2)
- Mobile browsers (limited)

### 📋 Requirements:
- Modern web browser
- JavaScript enabled
- Stable internet connection (for initial load)

## 🔄 Updates and Support

### Getting Updates:
```bash
cd "/home/adminlinux/2d FEA"
git pull  # If using git
# Or download new version
```

### For Advanced Users:
- Full Linux version with DOLFINx: Use `python src/main.py`
- Desktop GUI version: Available on Linux/WSL2
- Jupyter notebook: `2D_FEA_Colab.ipynb`

## 🌟 Tips for Best Results

1. **Start Simple**: Try predefined shapes first
2. **Regular Shapes**: Work better with analytical methods
3. **Reasonable Materials**: Use realistic material properties
4. **Save Work**: Export JSON files for later reference
5. **Browser Zoom**: Use 100% zoom for best interaction

## 📞 Support

If you encounter issues:
1. Check browser console for errors (F12)
2. Try refreshing the page
3. Use predefined shapes to verify functionality
4. Check material property values are reasonable

---

🎉 **Enjoy using the 2D FEA Web App!** 

This tool makes finite element analysis accessible to anyone with a web browser, perfect for education, quick calculations, and sharing with colleagues.
