@echo off
echo ================================================
echo 2D FEA Torsion Analysis - WSL2 Launcher
echo ================================================
echo.
echo Starting application in WSL2...
echo Make sure VcXsrv or X410 is running for GUI display.
echo.

REM Launch the application in WSL2
wsl -d Ubuntu -e bash -c "cd '/home/adminlinux/2d FEA' && conda activate fenics-core && python src/main.py"

echo.
echo Application closed.
pause
