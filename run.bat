@echo OFF
echo.
echo =================================================================
echo  Configuring environment and starting 2D FEA Application...
echo =================================================================
echo.

:: This script sets the necessary environment variables and ensures
:: the correct Python interpreter from the conda environment is used.

:: Ensure the user is in the correct environment first.
IF "%CONDA_DEFAULT_ENV%" NEQ "2d-fea-app" (
    echo ERROR: This script must be run from the '2d-fea-app' conda environment.
    echo Please run 'conda activate 2d-fea-app' first.
    pause
    exit /b 1
)

echo Activating environment settings...

:: 1. Set FI_PROVIDER to 'tcp'.
set FI_PROVIDER=tcp

:: 2. Set the Qt Plugin Path.
set QT_PLUGIN_PATH=%CONDA_PREFIX%\Lib\site-packages\PySide6\plugins

:: 3. Run the Python application using the explicit path from the environment.
echo Starting Python script with interpreter: %CONDA_PREFIX%\python.exe
"%CONDA_PREFIX%\python.exe" src/main.py

echo.
echo Application closed. Press any key to exit the terminal.
pause > nul