# Streamlit Cloud Deployment Guide

## 🚀 Deploy to Streamlit Cloud

This 2D FEA application is ready for deployment on Streamlit Cloud!

### 📋 Prerequisites
- GitHub repository (✅ already have this)
- Streamlit Cloud account (free)

### 🌐 Deployment Steps

1. **Go to Streamlit Cloud**
   - Visit: https://share.streamlit.io/
   - Sign in with your GitHub account

2. **Create New App**
   - Click "New app"
   - Select this repository: `YachtEngineeringservices/2d-FEA`
   - Set main file path: `src/web_app.py`
   - Set requirements file: `requirements-streamlit.txt`

3. **Deploy**
   - Click "Deploy"
   - Wait for automatic deployment (2-3 minutes)

### 📝 Configuration Files

- **`requirements-streamlit.txt`** - Python dependencies for cloud
- **`.streamlit/config.toml`** - Streamlit configuration  
- **`packages.txt`** - System dependencies (if needed)
- **`src/web_app.py`** - Main application file

### 🔗 Expected URL
Your app will be available at:
```
https://yacht-engineering-2d-fea.streamlit.app/
```

### ✨ Features in Web Version

- ✅ Interactive geometry input
- ✅ Real-time visualization
- ✅ Analytical torsional calculations  
- ✅ Stress and displacement analysis
- ✅ Results export (JSON)
- ✅ Mobile-friendly interface

### 🆚 Web vs Desktop Version

| Feature | Web Version | Desktop Version |
|---------|-------------|-----------------|
| **Accessibility** | Any browser | Windows/Linux only |
| **Installation** | None required | Download & extract |
| **Solver** | Analytical (fast) | DOLFINx (advanced) |
| **Mesh generation** | Simplified | Full GMSH |
| **File I/O** | JSON export | Full project files |
| **Performance** | Good for most cases | High-precision FEA |

### 🛠️ Local Development

To run locally:
```bash
pip install -r requirements-streamlit.txt
streamlit run src/web_app.py
```

### 📞 Support

- **Issues**: [GitHub Issues](https://github.com/YachtEngineeringservices/2d-FEA/issues)
- **Documentation**: [GitHub README](https://github.com/YachtEngineeringservices/2d-FEA)
- **Desktop Version**: [Releases](https://github.com/YachtEngineeringservices/2d-FEA/releases)
