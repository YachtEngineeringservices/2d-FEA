#!/bin/bash
# 2D FEA Application Launcher for WSL
# This script activates the correct conda environment and launches the application

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔧 2D FEA Application Launcher${NC}"
echo -e "${BLUE}=================================${NC}"

# Check if we're in WSL
if grep -qi microsoft /proc/version; then
    echo -e "${GREEN}✓ Running in WSL${NC}"
    
    # Check if X11 forwarding is available
    if [ -z "$DISPLAY" ]; then
        echo -e "${BLUE}Setting up X11 display...${NC}"
        export DISPLAY=:0.0
    fi
    
    # Check if X server is running on Windows
    if ! timeout 2 xset q &>/dev/null; then
        echo -e "${RED}❌ X11 server not detected on Windows${NC}"
        echo -e "${BLUE}Please install and start an X11 server on Windows:${NC}"
        echo "  - VcXsrv (recommended): https://sourceforge.net/projects/vcxsrv/"
        echo "  - Xming: https://sourceforge.net/projects/xming/"
        echo ""
        echo -e "${BLUE}After installation, start the X server with these settings:${NC}"
        echo "  - Display number: 0"
        echo "  - Disable access control: ✓"
        echo "  - Additional parameters: -ac"
        echo ""
        read -p "Press Enter to continue anyway (app may not display)..."
    else
        echo -e "${GREEN}✓ X11 server detected${NC}"
    fi
else
    echo -e "${GREEN}✓ Running on native Linux${NC}"
fi

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}❌ Conda not found. Please install Miniconda or Anaconda${NC}"
    exit 1
fi

# Initialize conda for bash if needed
if [[ ! "$PATH" == *"conda"* ]]; then
    echo -e "${BLUE}Initializing conda...${NC}"
    eval "$(conda shell.bash hook)"
fi

# Activate the fenics-core environment
echo -e "${BLUE}Activating fenics-core environment...${NC}"
conda activate fenics-core

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to activate fenics-core environment${NC}"
    echo -e "${BLUE}Creating environment...${NC}"
    conda create -n fenics-core python=3.11 -y
    conda activate fenics-core
    
    echo -e "${BLUE}Installing dependencies...${NC}"
    conda install -c conda-forge fenics-dolfinx matplotlib pyside6 -y
fi

# Navigate to the application directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${GREEN}✓ Environment ready${NC}"
echo -e "${BLUE}Launching 2D FEA Application...${NC}"
echo ""

# Launch the application
python src/main.py

echo -e "${BLUE}Application closed.${NC}"
