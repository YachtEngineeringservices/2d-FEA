# 2D FEA Application - Launch Instructions

## Quick Launch Options

### Option 1: Simple Script Launch (Recommended)
```bash
./launch_fea.sh
```

### Option 2: Create Terminal Alias
```bash
./setup_alias.sh
```
Then use:
```bash
fea
```

### Option 3: Direct Python Launch
```bash
conda activate fenics-core
python src/main.py
```

## Windows WSL Setup

### Prerequisites
1. **Install WSL2** with Ubuntu from Microsoft Store
2. **Install X11 Server** on Windows:
   - Download VcXsrv: https://sourceforge.net/projects/vcxsrv/
   - Install and run with these settings:
     - Display number: 0
     - ✓ Disable access control
     - Additional parameters: `-ac`

### Launch from Windows
1. **Double-click** `launch_fea_windows.bat`
2. **Or from Command Prompt/PowerShell:**
   ```cmd
   launch_fea_windows.bat
   ```

### WSL Terminal Launch
```bash
# In WSL terminal
cd "/mnt/c/path/to/2d FEA"  # Adjust path as needed
./launch_fea.sh
```

## Linux Desktop Setup

### Create Desktop Shortcut
1. Copy `2d-fea.desktop` to `~/Desktop/`
2. Make executable: `chmod +x ~/Desktop/2d-fea.desktop`
3. Double-click to launch

### Add to Applications Menu
```bash
cp 2d-fea.desktop ~/.local/share/applications/
```

## Troubleshooting

### X11 Display Issues (WSL)
If you get "cannot connect to X server" errors:
1. Make sure X11 server (VcXsrv) is running on Windows
2. Check display variable: `echo $DISPLAY`
3. Test X11: `xeyes` (install with `sudo apt install x11-apps`)

### Conda Environment Issues
If fenics-core environment doesn't exist:
```bash
conda create -n fenics-core python=3.11
conda activate fenics-core
conda install -c conda-forge fenics-dolfinx matplotlib pyside6
```

### Missing Dependencies
```bash
conda activate fenics-core
conda install -c conda-forge fenics-dolfinx matplotlib pyside6 numpy scipy
```

## Features
- ✅ Interactive geometry creation (point-and-click)
- ✅ Tabbed interface (Outer Shape / Inner Hole)
- ✅ Real-time mesh generation (GMSH)
- ✅ Professional FEA solver (DOLFINx)
- ✅ Stress visualization and analysis
- ✅ Material property configuration
- ✅ Results export and visualization

## System Requirements
- Linux or Windows with WSL2
- Python 3.11+
- Conda/Miniconda
- X11 server (for WSL on Windows)
- 4GB+ RAM recommended
