@echo off
REM 2D FEA Application Launcher for Windows (via WSL)
REM This batch file launches the FEA app in WSL from Windows

echo.
echo ====================================
echo  2D FEA Application Launcher
echo ====================================
echo.

REM Check if WSL is available
wsl --status >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: WSL is not installed or not running
    echo Please install WSL2 and Ubuntu from Microsoft Store
    pause
    exit /b 1
)

echo Starting 2D FEA Application in WSL...
echo.

REM Launch the application via WSL
wsl -d Ubuntu bash -c "cd '/home/%USERNAME%/2d FEA' && ./launch_fea.sh"

echo.
echo Application closed.
pause
