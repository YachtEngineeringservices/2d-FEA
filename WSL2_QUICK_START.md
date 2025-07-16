# Running 2D FEA in WSL2 - Quick Setup Guide

Since you're already using WSL2, you can run the Linux executable directly with a Windows GUI! This is actually the best approach as you get full FEA functionality.

## Current Status
✅ WSL2 is set up  
✅ X11 forwarding is configured (DISPLAY=:0)  
✅ Linux executables are built and ready  

## Quick Start

### Option 1: Direct Launch (Simple)
```bash
cd "/home/adminlinux/2d FEA/dist"
./2D_FEA_Simple
```

### Option 2: Use Launcher Script
```bash
cd "/home/adminlinux/2d FEA"
./run_2d_fea_wsl.sh
```

## If You Get Errors

The executable might have some library path issues in WSL2. If you see PETSc or library errors, try:

### Fix 1: Run from Source Instead
```bash
cd "/home/adminlinux/2d FEA"
conda activate fenics-core
python src/main.py
```

### Fix 2: Environment Variables
```bash
export PETSC_DIR=""
export PETSC_ARCH=""
cd dist
./2D_FEA_Simple
```

### Fix 3: X11 Server Setup

If the GUI doesn't appear:

1. **Install X11 Server on Windows:**
   - Download VcXsrv (free): https://sourceforge.net/projects/vcxsrv/
   - Or install X410 from Microsoft Store (paid)

2. **Configure VcXsrv:**
   - Start XLaunch
   - Choose "Multiple windows"
   - Display number: 0
   - ✅ Check "Disable access control"
   - ✅ Check "Native opengl"

3. **Test X11 in WSL2:**
   ```bash
   sudo apt update
   sudo apt install x11-apps
   xcalc  # Should open calculator
   ```

## Best Approach for WSL2

Since you already have WSL2 set up, I recommend running from source:

```bash
# In WSL2 terminal:
cd "/home/adminlinux/2d FEA"
conda activate fenics-core
python src/main.py
```

This gives you:
- ✅ Full DOLFINx FEA functionality
- ✅ All features working
- ✅ No PyInstaller bundle issues
- ✅ Windows GUI through X11 forwarding

## Troubleshooting

### If GUI doesn't show:
1. Check VcXsrv is running on Windows
2. Verify DISPLAY variable: `echo $DISPLAY`
3. Test with simple app: `xcalc`

### If libraries missing:
1. Activate conda environment: `conda activate fenics-core`
2. Install missing packages: `pip install [package_name]`

### If PETSc errors:
1. Run from source instead of executable
2. Clear environment: `unset PETSC_DIR PETSC_ARCH`

## Windows Integration

You can create a Windows shortcut to launch the WSL2 application:

1. **Create .bat file on Windows desktop:**
```batch
@echo off
wsl -d Ubuntu -e bash -c "cd '/home/adminlinux/2d FEA' && conda activate fenics-core && python src/main.py"
```

2. **Save as "2D_FEA.bat"**
3. **Double-click to run**

This gives you the best of both worlds: full Linux FEA functionality with Windows integration!
