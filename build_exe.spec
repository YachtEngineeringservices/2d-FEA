# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect data files for various packages
gmsh_datas = collect_data_files('gmsh')
dolfinx_datas = collect_data_files('dolfinx')
pyvista_datas = collect_data_files('pyvista')
matplotlib_datas = collect_data_files('matplotlib')

# Collect all submodules
hidden_imports = []
hidden_imports.extend(collect_submodules('dolfinx'))
hidden_imports.extend(collect_submodules('gmsh'))
hidden_imports.extend(collect_submodules('pyvista'))
hidden_imports.extend(collect_submodules('matplotlib'))
hidden_imports.extend(collect_submodules('meshio'))
hidden_imports.extend(collect_submodules('ufl'))
hidden_imports.extend(collect_submodules('PySide6'))

# Additional hidden imports that might be needed
hidden_imports.extend([
    'numpy',
    'numpy.core',
    'numpy.core._methods',
    'numpy.lib.format',
    'mpi4py',
    'petsc4py',
    'vtk',
    'logging',
    'json',
    'os',
    'sys'
])

block_cipher = None

a = Analysis(
    ['src/main.py'],
    pathex=['.'],
    binaries=[],
    datas=gmsh_datas + dolfinx_datas + pyvista_datas + matplotlib_datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='2D_FEA_Torsion_Analysis',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Set to False for GUI application
    disable_windowed_traceback=False,
    icon=None,  # Add path to .ico file if you have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='2D_FEA_Torsion_Analysis',
)
