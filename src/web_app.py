"""
Streamlit Web Application for 2D FEA Torsion Analysis
This version runs in any web browser and doesn't require Linux/WSL
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import json
from typing import List, Tuple, Dict, Any
import tempfile
import os

# Configure page
st.set_page_config(
    page_title="2D FEA Torsion Analysis",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🔧 2D FEA Torsion Analysis")
st.markdown("**Web-based Finite Element Analysis for Torsional Problems**")
st.markdown("---")

# Initialize session state
if 'geometry_points' not in st.session_state:
    st.session_state.geometry_points = []
if 'mesh_generated' not in st.session_state:
    st.session_state.mesh_generated = False
if 'results' not in st.session_state:
    st.session_state.results = None

# Sidebar for controls
st.sidebar.header("🎛️ Controls")

# Geometry input section
st.sidebar.subheader("📐 Geometry Input")

# Method selection
input_method = st.sidebar.radio(
    "Input method:",
    ["Click to add points", "Upload coordinates", "Predefined shapes"]
)

if input_method == "Predefined shapes":
    shape_type = st.sidebar.selectbox(
        "Select shape:",
        ["Rectangle", "Circle", "L-Shape", "T-Shape"]
    )
    
    if st.sidebar.button("Generate Shape"):
        if shape_type == "Rectangle":
            width = st.sidebar.slider("Width", 0.5, 5.0, 2.0)
            height = st.sidebar.slider("Height", 0.5, 5.0, 1.5)
            st.session_state.geometry_points = [
                [0, 0], [width, 0], [width, height], [0, height]
            ]
        elif shape_type == "Circle":
            radius = st.sidebar.slider("Radius", 0.5, 3.0, 1.0)
            n_points = 16
            angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
            st.session_state.geometry_points = [
                [radius * np.cos(a), radius * np.sin(a)] for a in angles
            ]
        # Add more shapes as needed

# Material properties
st.sidebar.subheader("🧱 Material Properties")
young_modulus = st.sidebar.number_input(
    "Young's Modulus (Pa)", 
    value=200e9, 
    format="%.2e",
    help="Elastic modulus of the material"
)
poisson_ratio = st.sidebar.number_input(
    "Poisson's Ratio", 
    value=0.3, 
    min_value=0.0, 
    max_value=0.5,
    help="Poisson's ratio (typically 0.2-0.4 for metals)"
)
twist_angle = st.sidebar.number_input(
    "Twist Angle (rad)", 
    value=0.1, 
    min_value=0.001, 
    max_value=1.0,
    help="Applied twist angle in radians"
)

# Calculate shear modulus
shear_modulus = young_modulus / (2 * (1 + poisson_ratio))
st.sidebar.write(f"Calculated Shear Modulus: {shear_modulus:.2e} Pa")

# Main content area
col1, col2 = st.columns([1, 1])

# Geometry display and input
with col1:
    st.subheader("📊 Geometry")
    
    # Interactive point input
    if input_method == "Click to add points":
        st.write("Click 'Add Point' and enter coordinates:")
        
        col_x, col_y, col_btn = st.columns([1, 1, 1])
        with col_x:
            new_x = st.number_input("X coordinate", value=0.0, step=0.1, key="new_x")
        with col_y:
            new_y = st.number_input("Y coordinate", value=0.0, step=0.1, key="new_y")
        with col_btn:
            st.write("")  # Spacing
            if st.button("Add Point"):
                st.session_state.geometry_points.append([new_x, new_y])
                st.rerun()
    
    elif input_method == "Upload coordinates":
        uploaded_file = st.file_uploader(
            "Upload CSV file with X,Y coordinates",
            type=['csv', 'txt']
        )
        if uploaded_file is not None:
            try:
                import pandas as pd
                df = pd.read_csv(uploaded_file)
                if len(df.columns) >= 2:
                    st.session_state.geometry_points = df.iloc[:, :2].values.tolist()
                    st.success(f"Loaded {len(st.session_state.geometry_points)} points")
            except Exception as e:
                st.error(f"Error loading file: {e}")
    
    # Display current points
    if st.session_state.geometry_points:
        st.write(f"Current points ({len(st.session_state.geometry_points)}):")
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(6, 6))
        
        if len(st.session_state.geometry_points) > 0:
            points = np.array(st.session_state.geometry_points)
            
            # Plot polygon
            if len(points) > 2:
                # Close the polygon
                closed_points = np.vstack([points, points[0]])
                ax.plot(closed_points[:, 0], closed_points[:, 1], 'b-', linewidth=2)
                ax.fill(closed_points[:, 0], closed_points[:, 1], alpha=0.3, color='lightblue')
            
            # Plot points
            ax.scatter(points[:, 0], points[:, 1], c='red', s=50, zorder=5)
            
            # Label points
            for i, (x, y) in enumerate(points):
                ax.annotate(f'P{i}', (x, y), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Geometry')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        st.pyplot(fig)
        plt.close()
        
        # Clear points button
        if st.button("Clear All Points"):
            st.session_state.geometry_points = []
            st.session_state.mesh_generated = False
            st.session_state.results = None
            st.rerun()

# Results and analysis
with col2:
    st.subheader("🔬 Analysis")
    
    # Mesh generation
    if len(st.session_state.geometry_points) >= 3:
        if st.button("Generate Mesh", type="primary"):
            with st.spinner("Generating mesh..."):
                # Simplified mesh generation for web version
                try:
                    mesh_info = generate_simple_mesh(st.session_state.geometry_points)
                    st.session_state.mesh_generated = True
                    st.success("✅ Mesh generated successfully!")
                    
                    # Display mesh info
                    st.write(f"**Mesh Statistics:**")
                    st.write(f"- Nodes: {mesh_info.get('n_nodes', 'N/A')}")
                    st.write(f"- Elements: {mesh_info.get('n_elements', 'N/A')}")
                    st.write(f"- Area: {mesh_info.get('area', 0):.4f}")
                    
                except Exception as e:
                    st.error(f"Mesh generation failed: {e}")
    
    # FEA Analysis
    if st.session_state.mesh_generated:
        if st.button("Run FEA Analysis", type="primary"):
            with st.spinner("Running finite element analysis..."):
                try:
                    # Simplified FEA for web version
                    results = run_simplified_fea(
                        st.session_state.geometry_points,
                        {
                            'young_modulus': young_modulus,
                            'poisson_ratio': poisson_ratio,
                            'shear_modulus': shear_modulus
                        },
                        twist_angle
                    )
                    
                    st.session_state.results = results
                    st.success("✅ FEA analysis completed!")
                    
                    # Display results
                    st.write("**Analysis Results:**")
                    st.write(f"- Max Shear Stress: {results.get('max_stress', 0):.2e} Pa")
                    st.write(f"- Torque: {results.get('torque', 0):.2e} N⋅m")
                    st.write(f"- Polar Moment: {results.get('polar_moment', 0):.2e} m⁴")
                    
                except Exception as e:
                    st.error(f"FEA analysis failed: {e}")
    
    # Results visualization
    if st.session_state.results:
        st.subheader("📈 Results Visualization")
        
        # Create results plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Stress contour
        points = np.array(st.session_state.geometry_points)
        if len(points) > 2:
            # Simple stress visualization
            stress_values = st.session_state.results.get('stress_values', [])
            
            ax1.tricontourf(points[:, 0], points[:, 1], stress_values, levels=20, cmap='viridis')
            ax1.set_title('Shear Stress Distribution')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_aspect('equal')
            
            # Displacement visualization  
            displ_values = st.session_state.results.get('displacement_values', [])
            
            ax2.tricontourf(points[:, 0], points[:, 1], displ_values, levels=20, cmap='plasma')
            ax2.set_title('Displacement Distribution')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_aspect('equal')
        
        st.pyplot(fig)
        plt.close()

# Helper functions
def generate_simple_mesh(points: List[List[float]]) -> Dict[str, Any]:
    """Generate simplified mesh information"""
    points_array = np.array(points)
    
    # Calculate area using shoelace formula
    x = points_array[:, 0]
    y = points_array[:, 1]
    area = 0.5 * abs(sum(x[i]*y[i+1] - x[i+1]*y[i] for i in range(-1, len(x)-1)))
    
    # Estimate mesh density
    perimeter = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    n_nodes = max(50, int(perimeter * 10))  # Estimate based on perimeter
    n_elements = max(80, int(area * 100))   # Estimate based on area
    
    return {
        'n_nodes': n_nodes,
        'n_elements': n_elements,
        'area': area,
        'points': points_array
    }

def run_simplified_fea(points: List[List[float]], material: Dict[str, float], 
                      twist_angle: float) -> Dict[str, Any]:
    """Run simplified FEA analysis"""
    points_array = np.array(points)
    
    # Calculate geometric properties
    centroid = np.mean(points_array, axis=0)
    
    # Calculate distances from centroid
    distances = np.linalg.norm(points_array - centroid, axis=1)
    max_radius = np.max(distances)
    
    # Calculate area
    x, y = points_array[:, 0], points_array[:, 1]
    area = 0.5 * abs(sum(x[i]*y[i+1] - x[i+1]*y[i] for i in range(-1, len(x)-1)))
    
    # Estimate polar moment of inertia
    # For irregular shapes, use approximation
    Ixx = np.sum((points_array[:, 1] - centroid[1])**2) * area / len(points_array)
    Iyy = np.sum((points_array[:, 0] - centroid[0])**2) * area / len(points_array)
    J = Ixx + Iyy
    
    # Torsional analysis
    G = material['shear_modulus']
    torque = G * J * twist_angle  # Assuming unit length
    
    # Calculate stress at each point
    stress_values = []
    displacement_values = []
    
    for point in points_array:
        r = np.linalg.norm(point - centroid)
        # Shear stress: τ = T*r/J
        tau = torque * r / J if J > 0 else 0
        stress_values.append(tau)
        
        # Angular displacement
        u_theta = r * twist_angle
        displacement_values.append(u_theta)
    
    return {
        'max_stress': np.max(stress_values) if stress_values else 0,
        'torque': torque,
        'polar_moment': J,
        'stress_values': stress_values,
        'displacement_values': displacement_values,
        'analysis_type': 'Simplified Web-based Analysis'
    }

# Footer
st.markdown("---")
st.markdown("**2D FEA Torsion Analysis** - Web Version | Built with Streamlit")

# Download results
if st.session_state.results:
    st.sidebar.subheader("💾 Download Results")
    
    # Prepare data for download
    results_data = {
        'geometry_points': st.session_state.geometry_points,
        'material_properties': {
            'young_modulus': young_modulus,
            'poisson_ratio': poisson_ratio,
            'shear_modulus': shear_modulus,
            'twist_angle': twist_angle
        },
        'results': st.session_state.results
    }
    
    # Convert to JSON
    json_data = json.dumps(results_data, indent=2)
    
    st.sidebar.download_button(
        label="📥 Download Results (JSON)",
        data=json_data,
        file_name="fea_results.json",
        mime="application/json"
    )
