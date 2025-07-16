#!/bin/bash
# Build script for creating Windows executable of 2D FEA application

echo "Building 2D FEA Torsional Analysis Windows Executable"
echo "======================================================="

# Install PyInstaller if not already installed
echo "Installing PyInstaller..."
pip install pyinstaller

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/
rm -rf dist/

# Create the executable using the spec file
echo "Building executable..."
pyinstaller build_exe.spec

# Check if build was successful
if [ -d "dist/2D_FEA_Torsion_Analysis" ]; then
    echo ""
    echo "Build completed successfully!"
    echo "==============================="
    echo "Executable location: dist/2D_FEA_Torsion_Analysis/"
    echo "Main executable: dist/2D_FEA_Torsion_Analysis/2D_FEA_Torsion_Analysis.exe"
    echo ""
    echo "To distribute:"
    echo "1. Copy the entire 'dist/2D_FEA_Torsion_Analysis' folder"
    echo "2. The folder contains all dependencies needed to run on Windows"
    echo "3. Run '2D_FEA_Torsion_Analysis.exe' to start the application"
    
    # Create a zip file for easy distribution
    if command -v zip &> /dev/null; then
        echo ""
        echo "Creating distribution zip file..."
        cd dist
        zip -r "2D_FEA_Torsion_Analysis_Windows.zip" "2D_FEA_Torsion_Analysis/"
        cd ..
        echo "Distribution zip created: dist/2D_FEA_Torsion_Analysis_Windows.zip"
    fi
else
    echo ""
    echo "Build failed! Check the output above for errors."
    exit 1
fi
