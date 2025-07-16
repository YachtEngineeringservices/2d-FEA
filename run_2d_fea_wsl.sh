#!/bin/bash

# 2D FEA Application Launcher for WSL2
# This script runs the Linux executable with proper X11 forwarding

echo "=================================================="
echo "2D FEA Torsion Analysis - WSL2 Launcher"
echo "=================================================="
echo

# Check if X11 forwarding is working
if [ -z "$DISPLAY" ]; then
    echo "Setting up X11 display..."
    export DISPLAY=:0.0
fi

echo "Display: $DISPLAY"

# Check if X11 server is accessible
echo "Testing X11 connection..."
if command -v xset >/dev/null 2>&1; then
    if xset q >/dev/null 2>&1; then
        echo "✓ X11 connection working"
    else
        echo "⚠ X11 server not responding"
        echo "Make sure VcXsrv or X410 is running on Windows"
        echo
        echo "To fix:"
        echo "1. Install VcXsrv (free) or X410 (Microsoft Store)"
        echo "2. Start the X server with 'Disable access control' enabled"
        echo "3. Run this script again"
        echo
        read -p "Press Enter to continue anyway or Ctrl+C to exit..."
    fi
else
    echo "Installing X11 utilities..."
    sudo apt update >/dev/null 2>&1
    sudo apt install -y x11-utils >/dev/null 2>&1
fi

echo

# Change to the directory containing the executable
cd "$(dirname "$0")/dist"

echo "Available executables:"
echo "1. 2D_FEA_Simple (single file, recommended)"
echo "2. 2D_FEA_Torsion_Analysis (directory version)"
echo

# Ask user which version to run
read -p "Choose version (1 or 2, default 1): " choice

case $choice in
    2)
        echo "Starting directory version..."
        if [ -f "2D_FEA_Torsion_Analysis/2D_FEA_Torsion_Analysis" ]; then
            cd 2D_FEA_Torsion_Analysis
            ./2D_FEA_Torsion_Analysis
        else
            echo "Error: Directory version not found!"
            echo "Using simple version instead..."
            cd ..
            ./2D_FEA_Simple
        fi
        ;;
    *)
        echo "Starting simple version..."
        if [ -f "2D_FEA_Simple" ]; then
            ./2D_FEA_Simple
        else
            echo "Error: Simple version not found!"
            echo "Please make sure you're in the correct directory."
            ls -la
        fi
        ;;
esac

echo
echo "Application closed."
