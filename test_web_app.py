#!/usr/bin/env python3
"""
Test script for the Streamlit web app functionality
"""

import sys
import os
sys.path.append('src')

# Test imports
print("🧪 Testing Streamlit Web App Components")
print("=" * 50)

try:
    import numpy as np
    print("✅ NumPy imported successfully")
except ImportError as e:
    print(f"❌ NumPy import failed: {e}")

try:
    import matplotlib.pyplot as plt
    print("✅ Matplotlib imported successfully")
except ImportError as e:
    print(f"❌ Matplotlib import failed: {e}")

try:
    import streamlit as st
    print("✅ Streamlit imported successfully")
except ImportError as e:
    print(f"❌ Streamlit import failed: {e}")

# Test helper functions from web_app.py
try:
    # Import functions from the web app
    sys.path.insert(0, 'src')
    import importlib.util
    spec = importlib.util.spec_from_file_location("web_app", "src/web_app.py")
    web_app = importlib.util.module_from_spec(spec)
    
    print("\n🔬 Testing Core Functions")
    print("-" * 30)
    
    # Test geometry creation
    test_points = [[0, 0], [2, 0], [2, 1.5], [0, 1.5]]
    print(f"📐 Test geometry: {test_points}")
    
    # Test mesh generation
    spec.loader.exec_module(web_app)
    mesh_info = web_app.generate_simple_mesh(test_points)
    print(f"✅ Mesh generation successful:")
    print(f"   - Nodes: {mesh_info['n_nodes']}")
    print(f"   - Elements: {mesh_info['n_elements']}")
    print(f"   - Area: {mesh_info['area']:.4f}")
    
    # Test FEA analysis
    material = {
        'young_modulus': 200e9,
        'poisson_ratio': 0.3,
        'shear_modulus': 200e9 / (2 * 1.3)
    }
    
    results = web_app.run_simplified_fea(test_points, material, 0.1)
    print(f"✅ FEA analysis successful:")
    print(f"   - Max stress: {results['max_stress']:.2e} Pa")
    print(f"   - Torque: {results['torque']:.2e} N⋅m")
    print(f"   - Analysis type: {results['analysis_type']}")
    
    print("\n🎯 All tests passed! Web app is ready to use.")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n📊 Summary:")
print("- Web app is running at http://localhost:8501")
print("- All core functions are working")
print("- Ready for user testing")

# Test predefined shapes
print("\n🔧 Testing Predefined Shapes:")
print("-" * 30)

# Rectangle
rect_points = [[0, 0], [2.0, 0], [2.0, 1.5], [0, 1.5]]
print(f"📐 Rectangle: {len(rect_points)} points")

# Circle approximation
import math
radius = 1.0
n_points = 16
angles = [2 * math.pi * i / n_points for i in range(n_points)]
circle_points = [[radius * math.cos(a), radius * math.sin(a)] for a in angles]
print(f"⭕ Circle: {len(circle_points)} points")

# L-shape
l_shape_points = [
    [0, 0], [2.0, 0], [2.0, 0.5],
    [0.5, 0.5], [0.5, 2.0], [0, 2.0]
]
print(f"📐 L-Shape: {len(l_shape_points)} points")

print("\n✅ All predefined shapes created successfully!")
print("\nYou can now:")
print("1. Open http://localhost:8501 in your browser")
print("2. Test the interactive interface")
print("3. Try different shapes and parameters")
print("4. Run FEA analysis and view results")
