# -*- mode: python ; coding: utf-8 -*-

"""
PyInstaller spec file for Windows executable
Use this file on a Windows machine to build the Windows .exe
"""

import os
import sys

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(SPEC))
sys.path.insert(0, current_dir)

block_cipher = None

# Analysis configuration
a = Analysis(
    ['src/main_windows.py'],  # Use Windows-specific main file
    pathex=[current_dir, os.path.join(current_dir, 'src')],
    binaries=[],
    datas=[
        # Include GUI files
        ('src/gui', 'gui'),
        # Include FEA files (Windows versions)
        ('src/fea/meshing.py', 'fea'),
        ('src/fea/solver_windows.py', 'fea'),  # Use Windows solver
        # Include any data files
        ('requirements.txt', '.'),
    ],
    hiddenimports=[
        # Core GUI imports
        'PySide6.QtCore',
        'PySide6.QtWidgets', 
        'PySide6.QtGui',
        
        # Matplotlib and backends
        'matplotlib',
        'matplotlib.pyplot',
        'matplotlib.backends.backend_qtagg',
        'matplotlib.figure',
        
        # Scientific computing
        'numpy',
        'numpy.core',
        'numpy.lib',
        
        # Meshing (Windows compatible)
        'gmsh',
        'meshio',
        'h5py',
        
        # Optional: SciPy if available
        'scipy',
        'scipy.spatial',
        
        # Project modules
        'gui.main_window',
        'gui.mpl_canvas',
        'fea.meshing',
        'fea.solver_windows',  # Windows-specific solver
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude Linux-specific packages
        'dolfinx',
        'fenics',
        'ufl',
        'ffcx',
        'basix',
        'petsc4py',
        'mpi4py',
        
        # Exclude unnecessary packages
        'tkinter',
        'turtle',
        'test',
        'tests',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Remove duplicates and sort
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Create executable
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='2D_FEA_Windows',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Windows GUI application
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add .ico file path if you have an icon
)

# Optional: Create directory distribution instead of single file
# Uncomment the following for directory distribution:

# coll = COLLECT(
#     exe,
#     a.binaries,
#     a.zipfiles,
#     a.datas,
#     strip=False,
#     upx=True,
#     upx_exclude=[],
#     name='2D_FEA_Windows_Dir'
# )
