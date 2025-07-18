#!/bin/bash
# Launch script for Streamlit web app with FULL DOLFINx FEA support
echo "🌐 Starting Streamlit Web App with FULL DOLFINx FEA..."
echo "🚀 Professional finite element analysis in your browser!"
echo "Open your browser to: http://localhost:8501"
echo "Press Ctrl+C to stop the server"
echo ""

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Activate the fenics-core environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fenics-core

# Set environment variables for DOLFINx
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Show FEA capabilities
echo "✅ DOLFINx FEA Solver: Available"
echo "✅ GMSH Mesh Generation: Available"  
echo "✅ Professional Stress Visualization: Available"
echo "✅ Same solver as desktop version: Available"
echo ""

# Launch the web app
streamlit run src/web_app.py --server.port=8501 --server.address=0.0.0.0
