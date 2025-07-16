@echo off
REM Build script for creating Windows executable of 2D FEA application

echo Building 2D FEA Torsional Analysis Windows Executable
echo =======================================================

REM Install PyInstaller if not already installed
echo Installing PyInstaller...
pip install pyinstaller

REM Clean previous builds
echo Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

REM Create the executable using the spec file
echo Building executable...
pyinstaller build_exe.spec

REM Check if build was successful
if exist "dist\2D_FEA_Torsion_Analysis" (
    echo.
    echo Build completed successfully!
    echo ==============================
    echo Executable location: dist\2D_FEA_Torsion_Analysis\
    echo Main executable: dist\2D_FEA_Torsion_Analysis\2D_FEA_Torsion_Analysis.exe
    echo.
    echo To distribute:
    echo 1. Copy the entire 'dist\2D_FEA_Torsion_Analysis' folder
    echo 2. The folder contains all dependencies needed to run on Windows
    echo 3. Run '2D_FEA_Torsion_Analysis.exe' to start the application
    
    REM Create a zip file for easy distribution if 7zip is available
    where 7z >nul 2>nul
    if %errorlevel% == 0 (
        echo.
        echo Creating distribution zip file...
        cd dist
        7z a "2D_FEA_Torsion_Analysis_Windows.zip" "2D_FEA_Torsion_Analysis\"
        cd ..
        echo Distribution zip created: dist\2D_FEA_Torsion_Analysis_Windows.zip
    )
) else (
    echo.
    echo Build failed! Check the output above for errors.
    pause
    exit /b 1
)

pause
