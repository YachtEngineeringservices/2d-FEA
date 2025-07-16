@echo OFF
echo This script will set up the Python environment for the 2D FEA application.
echo Please make sure you have Anaconda or Miniconda installed.
echo.
pause

echo [1/3] Creating Conda environment "2d-fea-app" with Python 3.10...
conda create --name 2d-fea-app python=3.10 -y
IF %ERRORLEVEL% NEQ 0 (
    echo Failed to create the conda environment.
    pause
    exit /b 1
)

echo.
echo [2/3] Installing packages into "2d-fea-app"...
echo This will take several minutes.

echo Installing scientific packages with Conda...
conda run -n 2d-fea-app conda install -c conda-forge -y fenics-dolfinx gmsh meshio matplotlib mpich
IF %ERRORLEVEL% NEQ 0 (
    echo Failed to install scientific packages with Conda.
    pause
    exit /b 1
)

echo Installing GUI package with Pip...
conda run -n 2d-fea-app pip install pyside6
IF %ERRORLEVEL% NEQ 0 (
    echo Failed to install PySide6 with Pip.
    pause
    exit /b 1
)

echo.
echo [3/3] Verifying PySide6 installation...
conda run -n 2d-fea-app pip show pyside6
IF %ERRORLEVEL% NEQ 0 (
    echo Verification failed. PySide6 may not have installed correctly.
    pause
    exit /b 1
)

echo.
echo ---
echo Environment setup is complete!
echo You can now run the application using:
echo   .\run.bat
echo ---
pause
