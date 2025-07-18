"""
Streamlit Web Application for 2D FEA Torsion Analysis
This version uses the same DOLFINx solver and GMSH mesher as the desktop version
Professional FEA analysis in your web browser
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import tempfile
import os
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap
import shutil

# Set matplotlib backend for cloud environment
import matplotlib
matplotlib.use('Agg')

# Import the same FEA components as desktop version
try:
    from fea import meshing, solver
    from fea.solver import solve_torsion
    FEA_AVAILABLE = True
    st.success("✅ Full FEA solver (DOLFINx) available")
except ImportError as e:
    st.error(f"❌ FEA solver not available: {e}")
    st.info("📝 Falling back to simplified calculations")
    FEA_AVAILABLE = False

# Fallback simplified solver for environments without DOLFINx
def run_simplified_fea(outer_points, inner_points, G, T, L):
    """
    Simplified torsional analysis fallback
    """
    st.warning("⚠️ Using simplified analytical solver - install DOLFINx for full FEA")
    
    try:
        # Calculate area using shoelace formula
        def polygon_area(points):
            if len(points) < 3:
                return 0.0
            points = np.array(points)
            x = points[:, 0]
            y = points[:, 1]
            return 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] for i in range(-1, len(x)-1)))
        
        # Calculate approximate polar moment
        def polar_moment_approx(outer_pts, inner_pts):
            outer_area = polygon_area(outer_pts)
            inner_area = polygon_area(inner_pts) if inner_pts else 0.0
            net_area = outer_area - inner_area
            
            if outer_pts:
                outer_pts = np.array(outer_pts)
                width = np.max(outer_pts[:, 0]) - np.min(outer_pts[:, 0])
                height = np.max(outer_pts[:, 1]) - np.min(outer_pts[:, 1])
                char_dim = (width + height) / 2
            else:
                char_dim = 1.0
            
            if inner_area == 0:
                J = net_area * char_dim**2 / 6
            else:
                J = net_area * char_dim**2 / 8
            
            return max(J, 1e-12)
        
        # Calculate geometry properties
        J = polar_moment_approx(outer_points, inner_points)
        k = G * J / L
        theta = T / k
        
        # Estimate stress at points
        if outer_points:
            outer_pts = np.array(outer_points)
            centroid = np.mean(outer_pts, axis=0)
            distances = np.sqrt(np.sum((outer_pts - centroid)**2, axis=1))
            r_max = np.max(distances)
            
            stress_at_points = []
            for point in outer_pts:
                r = np.sqrt(np.sum((point - centroid)**2))
                tau = T * r / J if J > 0 else 0
                stress_at_points.append(tau)
            
            inner_stress_at_points = []
            if inner_points:
                inner_pts = np.array(inner_points)
                for point in inner_pts:
                    r = np.sqrt(np.sum((point - centroid)**2))
                    tau = T * r / J if J > 0 else 0
                    inner_stress_at_points.append(tau)
        else:
            r_max = 1.0
            stress_at_points = []
            inner_stress_at_points = []
        
        tau_max = T * r_max / J
        
        return {
            'polar_moment': J,
            'stiffness': k,
            'twist_angle': theta,
            'max_shear_stress': tau_max,
            'outer_stress_values': stress_at_points,
            'inner_stress_values': inner_stress_at_points,
            'stress_field': None,  # No field data for simplified solver
            'mesh_points': outer_points,
            'success': True,
            'solver_type': 'simplified'
        }
        
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return {
            'polar_moment': 0,
            'stiffness': 0, 
            'twist_angle': 0,
            'max_shear_stress': 0,
            'outer_stress_values': [],
            'inner_stress_values': [],
            'stress_field': None,
            'mesh_points': [],
            'success': False,
            'solver_type': 'simplified'
        }

def run_full_fea(outer_points, inner_points, G, T, L, mesh_size=0.01):
    """
    Full FEA analysis using DOLFINx and GMSH (same as desktop version)
    """
    if not FEA_AVAILABLE:
        return run_simplified_fea(outer_points, inner_points, G, T, L)
    
    st.info("🔧 Running full FEA analysis with DOLFINx...")
    
    try:
        # Generate mesh using GMSH (same as desktop)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Generating mesh with GMSH...")
        progress_bar.progress(20)
        
        # Use the same meshing function as desktop
        output_dir = meshing.create_mesh(
            outer_points, inner_points, mesh_size
        )
        
        if not output_dir:
            st.error("❌ Mesh generation failed")
            return run_simplified_fea(outer_points, inner_points, G, T, L)
        
        progress_bar.progress(50)
        status_text.text("Solving with DOLFINx...")
        
        # Use the same solver as desktop
        J, k, theta, tau_max, tau_magnitude, V_mag = solve_torsion(
            output_dir, G, T, L
        )
        
        progress_bar.progress(80)
        status_text.text("Extracting results...")
        
        # Extract stress field data for visualization
        stress_field = None
        mesh_points = []
        
        try:
            # Extract stress values and mesh coordinates
            import dolfinx
            from dolfinx import mesh as dmesh
            
            # Read mesh for visualization
            mesh_file = os.path.join(output_dir, "domain.xdmf")
            if os.path.exists(mesh_file):
                with dolfinx.io.XDMFFile(mesh.comm, mesh_file, "r") as file:
                    domain = file.read_mesh()
                    
                # Extract mesh points
                mesh_points = domain.geometry.x
                
                # Extract stress values
                if tau_magnitude is not None:
                    stress_field = tau_magnitude.x.array
            
        except Exception as e:
            st.warning(f"Could not extract mesh visualization data: {e}")
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        # Calculate stress at boundary points for point visualization
        outer_stress_values = []
        inner_stress_values = []
        
        if outer_points:
            outer_pts = np.array(outer_points)
            centroid = np.mean(outer_pts, axis=0)
            
            for point in outer_pts:
                r = np.sqrt(np.sum((point - centroid)**2))
                tau = T * r / J if J > 0 else 0
                outer_stress_values.append(tau)
            
            if inner_points:
                inner_pts = np.array(inner_points)
                for point in inner_pts:
                    r = np.sqrt(np.sum((point - centroid)**2))
                    tau = T * r / J if J > 0 else 0
                    inner_stress_values.append(tau)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return {
            'polar_moment': J,
            'stiffness': k,
            'twist_angle': theta,
            'max_shear_stress': tau_max,
            'outer_stress_values': outer_stress_values,
            'inner_stress_values': inner_stress_values,
            'stress_field': stress_field,
            'mesh_points': mesh_points,
            'success': True,
            'solver_type': 'full_fea'
        }
        
    except Exception as e:
        st.error(f"FEA analysis error: {str(e)}")
        st.info("Falling back to simplified analysis...")
        return run_simplified_fea(outer_points, inner_points, G, T, L)

def parse_points_input(text):
    """
    Parse multiple coordinate input formats:
    - One pair per line: "0, 0\\n100, 0\\n100, 100"
    - Comma-separated: "0,0, 100,0, 100,100"
    - Space-separated: "0 0\\n100 0\\n100 100"
    - Mixed formats
    """
    points = []
    
    # Clean up the text
    text = text.strip().replace('\r', '')
    
    # Try to handle different formats
    if '\n' in text:
        # Line-by-line format
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try comma-separated first
            if ',' in line:
                parts = line.split(',')
                # Handle multiple pairs per line
                for i in range(0, len(parts)-1, 2):
                    try:
                        x = float(parts[i].strip())
                        y = float(parts[i+1].strip())
                        points.append([x, y])
                    except (ValueError, IndexError):
                        continue
            else:
                # Try space-separated
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        points.append([x, y])
                    except ValueError:
                        continue
    else:
        # Single line format
        if ',' in text:
            # Comma-separated values
            parts = text.split(',')
            parts = [p.strip() for p in parts if p.strip()]
            
            # Try to pair them up
            for i in range(0, len(parts)-1, 2):
                try:
                    x = float(parts[i])
                    y = float(parts[i+1])
                    points.append([x, y])
                except (ValueError, IndexError):
                    continue
        else:
            # Space-separated values
            parts = text.split()
            for i in range(0, len(parts)-1, 2):
                try:
                    x = float(parts[i])
                    y = float(parts[i+1])
                    points.append([x, y])
                except (ValueError, IndexError):
                    continue
    
    return points

# Configure page
st.set_page_config(
    page_title="2D FEA Torsion Analysis - Yacht Engineering Services",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'outer_points' not in st.session_state:
    st.session_state.outer_points = []
if 'inner_points' not in st.session_state:
    st.session_state.inner_points = []
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Outer Shape"
if 'results' not in st.session_state:
    st.session_state.results = None

# Header
st.title("🔧 2D FEA Torsion Analysis")
st.markdown("**Web-based Finite Element Analysis for Torsional Loading**")
st.markdown("*Interactive cross-section geometry editor with real-time visualization*")

# Create main layout with sidebar and main area
col1, col2 = st.columns([1, 2])

with col1:
    st.header("🎛️ Controls")
    
    # Geometry Tabs
    st.subheader("📐 Geometry Definition")
    geometry_tab = st.radio(
        "Select geometry to edit:",
        ["Outer Shape", "Inner Hole"],
        index=0 if st.session_state.active_tab == "Outer Shape" else 1,
        key="geometry_tab_selector"
    )
    st.session_state.active_tab = geometry_tab
    
    # Point input section
    if geometry_tab == "Outer Shape":
        current_points = st.session_state.outer_points
        st.write(f"**Outer Shape Points** ({len(current_points)} points)")
    else:
        current_points = st.session_state.inner_points  
        st.write(f"**Inner Hole Points** ({len(current_points)} points)")
    
    # Multi-point input controls
    st.write("**Add Multiple Points:**")
    st.write("*Paste coordinates as X,Y pairs (one per line or comma-separated)*")
    
    # Example formats
    with st.expander("📋 Input Format Examples"):
        st.code("""
Format 1 (one pair per line):
0, 0
100, 0
100, 100
0, 100

Format 2 (comma-separated):
0,0, 100,0, 100,100, 0,100

Format 3 (space-separated):
0 0
100 0
100 100
0 100
        """)
    
    points_text = st.text_area(
        "Enter coordinates:",
        height=100,
        placeholder="0, 0\n100, 0\n100, 100\n0, 100",
        key=f"points_input_{geometry_tab}"
    )
    
    col_add, col_replace = st.columns(2)
    
    with col_add:
        if st.button("➕ Add Points", type="primary"):
            if points_text.strip():
                try:
                    # Parse the input text
                    new_points = parse_points_input(points_text)
                    
                    if new_points:
                        if geometry_tab == "Outer Shape":
                            st.session_state.outer_points.extend(new_points)
                            st.success(f"Added {len(new_points)} points to Outer Shape")
                        else:
                            st.session_state.inner_points.extend(new_points)
                            st.success(f"Added {len(new_points)} points to Inner Hole")
                        st.rerun()
                    else:
                        st.error("No valid points found in input")
                except Exception as e:
                    st.error(f"Error parsing points: {str(e)}")
            else:
                st.warning("Please enter some coordinates")
    
    with col_replace:
        if st.button("🔄 Replace All", type="secondary"):
            if points_text.strip():
                try:
                    # Parse the input text
                    new_points = parse_points_input(points_text)
                    
                    if new_points:
                        if geometry_tab == "Outer Shape":
                            st.session_state.outer_points = new_points
                            st.success(f"Replaced with {len(new_points)} points for Outer Shape")
                        else:
                            st.session_state.inner_points = new_points
                            st.success(f"Replaced with {len(new_points)} points for Inner Hole")
                        st.rerun()
                    else:
                        st.error("No valid points found in input")
                except Exception as e:
                    st.error(f"Error parsing points: {str(e)}")
            else:
                st.warning("Please enter some coordinates")
    
    # Point list and editing
    if current_points:
        st.write("**Current Points:**")
        
        # Display points as editable table
        points_df = pd.DataFrame(current_points, columns=['X (mm)', 'Y (mm)'])
        points_df.index = points_df.index + 1  # Start from 1 instead of 0
        
        # Show the dataframe
        st.dataframe(points_df, use_container_width=True)
        
        # Point management buttons
        col_clear, col_remove = st.columns(2)
        with col_clear:
            if st.button("Clear All", type="secondary"):
                if geometry_tab == "Outer Shape":
                    st.session_state.outer_points = []
                else:
                    st.session_state.inner_points = []
                st.rerun()
        
        with col_remove:
            if len(current_points) > 0:
                remove_idx = st.selectbox("Remove point:", 
                                        range(1, len(current_points) + 1),
                                        format_func=lambda x: f"Point {x}")
                if st.button("Remove"):
                    if geometry_tab == "Outer Shape":
                        st.session_state.outer_points.pop(remove_idx - 1)
                    else:
                        st.session_state.inner_points.pop(remove_idx - 1)
                    st.rerun()
    
    # Quick shapes
    st.write("**Quick Shapes:**")
    
    shape_cols = st.columns(2)
    
    with shape_cols[0]:
        if st.button("🟫 Rectangle\n(100×50mm)", use_container_width=True):
            points = [[0, 0], [100, 0], [100, 50], [0, 50]]
            if geometry_tab == "Outer Shape":
                st.session_state.outer_points = points
            else:
                st.session_state.inner_points = points
            st.rerun()
        
        if st.button("🔺 Triangle\n(Equilateral)", use_container_width=True):
            import math
            side = 100
            height = side * math.sqrt(3) / 2
            points = [[0, 0], [side, 0], [side/2, height]]
            if geometry_tab == "Outer Shape":
                st.session_state.outer_points = points
            else:
                st.session_state.inner_points = points
            st.rerun()
    
    with shape_cols[1]:
        if st.button("⭕ Circle\n(r=50mm)", use_container_width=True):
            import math
            radius = 50
            n_points = 16
            angles = [i * 2 * math.pi / n_points for i in range(n_points)]
            points = [[radius * math.cos(a), radius * math.sin(a)] for a in angles]
            if geometry_tab == "Outer Shape":
                st.session_state.outer_points = points
            else:
                st.session_state.inner_points = points
            st.rerun()
        
        if st.button("📐 L-Shape\n(100×100×20mm)", use_container_width=True):
            # Standard L-section
            points = [[0, 0], [100, 0], [100, 20], [20, 20], [20, 100], [0, 100]]
            if geometry_tab == "Outer Shape":
                st.session_state.outer_points = points
            else:
                st.session_state.inner_points = points
            st.rerun()
    
    # Analysis Controls
    st.markdown("---")
    st.subheader("🔬 Analysis Controls")
    
    mesh_size = st.number_input("Mesh Size (mm):", value=10.0, min_value=0.1, max_value=50.0, format="%.1f")
    
    # Material Properties
    st.write("**Material Properties:**")
    shear_modulus = st.number_input("Shear Modulus G (MPa):", value=80000.0, format="%.1f")
    
    # Loading
    st.write("**Loading:**")
    applied_torque = st.number_input("Applied Torque T (N⋅m):", value=1000.0, format="%.1f")
    beam_length = st.number_input("Beam Length L (m):", value=2.0, format="%.2f")
    
    # Analysis button
    if st.button("🚀 Generate Mesh & Solve", type="primary"):
        if len(st.session_state.outer_points) < 3:
            st.error("❌ Please define at least 3 points for the Outer Shape.")
        elif len(st.session_state.inner_points) > 0 and len(st.session_state.inner_points) < 3:
            st.error("❌ If you define an Inner Hole, it must have at least 3 points.")
        else:
            with st.spinner("Running analysis..."):
                # Convert to meters and run full FEA analysis
                outer_points_m = [[p[0]/1000, p[1]/1000] for p in st.session_state.outer_points]
                inner_points_m = [[p[0]/1000, p[1]/1000] for p in st.session_state.inner_points]
                
                # Use full FEA solver (same as desktop version)
                results = run_full_fea(
                    outer_points_m, 
                    inner_points_m,
                    shear_modulus * 1e6,  # Convert MPa to Pa
                    applied_torque,
                    beam_length,
                    mesh_size / 1000  # Convert mm to m
                )
                
                st.session_state.results = results
                
                if results['success']:
                    if results['solver_type'] == 'full_fea':
                        st.success("✅ Full FEA analysis completed with DOLFINx!")
                    else:
                        st.warning("⚠️ Analysis completed with simplified solver")
                else:
                    st.error("❌ Analysis failed")
                st.rerun()
    
    # Results Display
    if st.session_state.results:
        st.markdown("---")
        st.subheader("📊 Results")
        
        results = st.session_state.results
        
        # Convert units for display
        J_mm4 = results['polar_moment'] * (1000**4)  # m^4 to mm^4
        k = results['stiffness']
        theta_deg = np.rad2deg(results['twist_angle'])
        tau_max_mpa = results['max_shear_stress'] / 1e6  # Pa to MPa
        
        st.metric("Torsional Constant J", f"{J_mm4:.2e} mm⁴")
        st.metric("Torsional Stiffness k", f"{k:.2e} N⋅m/rad")
        st.metric("Angle of Twist θ", f"{theta_deg:.4f}°")
        st.metric("Max Shear Stress τ_max", f"{tau_max_mpa:.2f} MPa")
        
        # Download results
        if st.button("💾 Download Results"):
            results_data = {
                'geometry': {
                    'outer_points': st.session_state.outer_points,
                    'inner_points': st.session_state.inner_points
                },
                'material': {
                    'shear_modulus_mpa': shear_modulus,
                    'applied_torque_nm': applied_torque,
                    'beam_length_m': beam_length
                },
                'results': {
                    'torsional_constant_mm4': float(J_mm4),
                    'torsional_stiffness_nm_per_rad': float(k),
                    'twist_angle_degrees': float(theta_deg),
                    'max_shear_stress_mpa': float(tau_max_mpa)
                }
            }
            
            json_str = json.dumps(results_data, indent=2)
            st.download_button(
                label="Download as JSON",
                data=json_str,
                file_name="fea_results.json",
                mime="application/json"
            )

with col2:
    # Switch between geometry and results view
    if st.session_state.results:
        view_tab = st.radio("View:", ["Geometry", "Stress Results"], horizontal=True)
    else:
        view_tab = "Geometry"
        
    if view_tab == "Geometry":
        st.header("📈 Cross-Section Geometry")
    else:
        st.header("🌡️ Stress Distribution")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Check if we should show stress results
    show_stress = (view_tab == "Stress Results" and 
                   st.session_state.results and 
                   st.session_state.results['success'])
    
    # Plot outer shape
    if len(st.session_state.outer_points) >= 3:
        outer_array = np.array(st.session_state.outer_points)
        
        if show_stress:
            # Plot with stress visualization
            results = st.session_state.results
            
            # Check if we have full FEA stress field data
            if (results['solver_type'] == 'full_fea' and 
                results['stress_field'] is not None and 
                results['mesh_points'] is not None):
                
                # Full FEA contour plot
                try:
                    mesh_points = results['mesh_points']
                    stress_field = results['stress_field']
                    
                    # Convert mesh points to mm for display
                    mesh_x = mesh_points[:, 0] * 1000
                    mesh_y = mesh_points[:, 1] * 1000
                    stress_mpa = stress_field / 1e6  # Convert to MPa
                    
                    # Create contour plot
                    contour = ax.tricontourf(mesh_x, mesh_y, stress_mpa, 
                                           levels=20, cmap='viridis', alpha=0.8)
                    
                    # Add colorbar
                    cbar = plt.colorbar(contour, ax=ax, label='Shear Stress (MPa)')
                    
                    # Add contour lines
                    ax.tricontour(mesh_x, mesh_y, stress_mpa, 
                                levels=10, colors='white', alpha=0.5, linewidths=0.5)
                    
                    # Plot polygon outline
                    outer_polygon = Polygon(outer_array, fill=False, edgecolor='white', 
                                          linewidth=3, label='Outer Shape')
                    ax.add_patch(outer_polygon)
                    
                    st.info("🎯 Showing full FEA stress field from DOLFINx mesh")
                    
                except Exception as e:
                    st.warning(f"Could not create FEA contour plot: {e}")
                    # Fall back to point-based visualization
                    show_simplified_stress = True
                    
            else:
                # Simplified stress visualization
                show_simplified_stress = True
                
            # Simplified stress visualization (for analytical solver or FEA fallback)
            if 'show_simplified_stress' in locals() and show_simplified_stress:
                stress_values = results['outer_stress_values']
                if stress_values:
                    stress_mpa = [s / 1e6 for s in stress_values]  # Convert to MPa
                    
                    # Create a filled contour plot for stress distribution
                    from scipy.interpolate import griddata
                    
                    # Create a grid for interpolation
                    x_min, x_max = np.min(outer_array[:, 0]) - 10, np.max(outer_array[:, 0]) + 10
                    y_min, y_max = np.min(outer_array[:, 1]) - 10, np.max(outer_array[:, 1]) + 10
                    
                    # Create grid
                    xi = np.linspace(x_min, x_max, 100)
                    yi = np.linspace(y_min, y_max, 100)
                    XI, YI = np.meshgrid(xi, yi)
                    
                    # Interpolate stress values
                    ZI = griddata((outer_array[:, 0], outer_array[:, 1]), stress_mpa, (XI, YI), 
                                method='cubic', fill_value=0)
                    
                    # Create a mask for the polygon area
                    from matplotlib.path import Path
                    polygon_path = Path(outer_array)
                    points = np.column_stack((XI.ravel(), YI.ravel()))
                    mask = polygon_path.contains_points(points).reshape(XI.shape)
                    
                    # Apply mask to stress data
                    ZI_masked = np.where(mask, ZI, np.nan)
                    
                    # Create filled contour plot
                    contour = ax.contourf(XI, YI, ZI_masked, levels=20, cmap='viridis', alpha=0.8)
                    
                    # Add colorbar
                    cbar = plt.colorbar(contour, ax=ax, label='Shear Stress (MPa)')
                    
                    # Plot polygon outline
                    outer_polygon = Polygon(outer_array, fill=False, edgecolor='white', linewidth=3)
                    ax.add_patch(outer_polygon)
                    
                    if results['solver_type'] == 'simplified':
                        st.info("📊 Showing interpolated stress distribution from analytical solver")
                    else:
                        st.info("📊 Showing stress distribution at boundary points")
            
            # Plot stress at boundary points with values
            if results['outer_stress_values']:
                stress_mpa = [s / 1e6 for s in results['outer_stress_values']]
                scatter = ax.scatter(outer_array[:, 0], outer_array[:, 1], 
                                   c=stress_mpa, s=150, cmap='viridis', 
                                   edgecolor='white', linewidth=2, zorder=5)
                
                # Label points with stress values
                for i, ((x, y), stress) in enumerate(zip(outer_array, stress_mpa)):
                    ax.annotate(f'{i+1}\n{stress:.1f}', (x, y), xytext=(5, 5), 
                               textcoords='offset points', fontsize=9, 
                               fontweight='bold', color='white',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))
        else:
            # Regular geometry view
            outer_polygon = Polygon(outer_array, alpha=0.3, facecolor='lightblue', 
                                   edgecolor='blue', linewidth=2, label='Outer Shape')
            ax.add_patch(outer_polygon)
            
            # Plot outer points
            ax.scatter(outer_array[:, 0], outer_array[:, 1], c='blue', s=100, zorder=5)
            
            # Label outer points
            for i, (x, y) in enumerate(outer_array):
                ax.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points',
                           fontsize=10, fontweight='bold', color='blue')
    
    # Plot inner hole
    if len(st.session_state.inner_points) >= 3:
        inner_array = np.array(st.session_state.inner_points)
        
        if show_stress:
            # Create inner hole outline (cutout for full FEA, stress points for simplified)
            results = st.session_state.results
            
            if (results['solver_type'] == 'full_fea' and 
                results['stress_field'] is not None):
                # For full FEA, just create a white hole to show the cutout
                inner_polygon = Polygon(inner_array, fill=True, facecolor='white', 
                                       edgecolor='white', linewidth=3, zorder=10)
                ax.add_patch(inner_polygon)
                
                # Add boundary outline
                inner_outline = Polygon(inner_array, fill=False, edgecolor='white', 
                                       linewidth=2, zorder=11)
                ax.add_patch(inner_outline)
                
            else:
                # Plot with stress coloring for simplified solver
                stress_values = results['inner_stress_values']
                if stress_values:  # Only if we have stress data
                    stress_mpa = [s / 1e6 for s in stress_values]  # Convert to MPa
                    
                    # Create inner hole outline (cutout)
                    inner_polygon = Polygon(inner_array, fill=True, facecolor='white', 
                                           edgecolor='white', linewidth=3, zorder=10)
                    ax.add_patch(inner_polygon)
                    
                    # Plot stress points for inner boundary
                    scatter_inner = ax.scatter(inner_array[:, 0], inner_array[:, 1], 
                                             c=stress_mpa, s=150, cmap='viridis', 
                                             edgecolor='white', linewidth=2, zorder=15)
                    
                    # Label points with stress values
                    for i, ((x, y), stress) in enumerate(zip(inner_array, stress_mpa)):
                        ax.annotate(f'{i+1}\n{stress:.1f}', (x, y), xytext=(5, 5), 
                                   textcoords='offset points', fontsize=9, 
                                   fontweight='bold', color='black',
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
                else:
                    # Just show the hole outline
                    inner_polygon = Polygon(inner_array, fill=True, facecolor='white', 
                                           edgecolor='white', linewidth=3, zorder=10)
                    ax.add_patch(inner_polygon)
        else:
            # Regular geometry view
            inner_polygon = Polygon(inner_array, alpha=0.8, facecolor='white', 
                                   edgecolor='red', linewidth=2, label='Inner Hole')
            ax.add_patch(inner_polygon)
            
            # Plot inner points
            ax.scatter(inner_array[:, 0], inner_array[:, 1], c='red', s=100, zorder=5)
            
            # Label inner points
            for i, (x, y) in enumerate(inner_array):
                ax.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points',
                           fontsize=10, fontweight='bold', color='red')
    
    # Plot individual points for active geometry (only in geometry view)
    if not show_stress:
        current_points = st.session_state.outer_points if st.session_state.active_tab == "Outer Shape" else st.session_state.inner_points
        color = 'blue' if st.session_state.active_tab == "Outer Shape" else 'red'
        
        if current_points and len(current_points) < 3:
            points_array = np.array(current_points)
            ax.scatter(points_array[:, 0], points_array[:, 1], c=color, s=100, zorder=5)
            for i, (x, y) in enumerate(points_array):
                ax.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points',
                           fontsize=10, fontweight='bold', color=color)
    
    # Formatting
    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    
    if show_stress:
        ax.set_title('Shear Stress Distribution\n(Values shown at each point)', fontsize=14)
    else:
        ax.set_title(f'Cross-Section Geometry\n{st.session_state.active_tab} (Enter coordinates in left panel)', fontsize=14)
    
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Set reasonable limits
    all_points = st.session_state.outer_points + st.session_state.inner_points
    if all_points:
        all_array = np.array(all_points)
        x_margin = max(10, (np.max(all_array[:, 0]) - np.min(all_array[:, 0])) * 0.1)
        y_margin = max(10, (np.max(all_array[:, 1]) - np.min(all_array[:, 1])) * 0.1)
        ax.set_xlim(np.min(all_array[:, 0]) - x_margin, np.max(all_array[:, 0]) + x_margin)
        ax.set_ylim(np.min(all_array[:, 1]) - y_margin, np.max(all_array[:, 1]) + y_margin)
    else:
        ax.set_xlim(-50, 150)
        ax.set_ylim(-50, 150)
    
    # Add legend if both shapes exist
    if len(st.session_state.outer_points) >= 3 or len(st.session_state.inner_points) >= 3:
        ax.legend()
    
    # Instructions
    if show_stress:
        results = st.session_state.results
        max_stress_mpa = results['max_shear_stress'] / 1e6
        
        if results['solver_type'] == 'full_fea':
            instructions = f"""
            **Full FEA Analysis Results (DOLFINx):**
            - Max shear stress: **{max_stress_mpa:.2f} MPa**
            - Professional finite element analysis with mesh generation
            - Stress field computed from DOLFINx solver
            - Contour plot shows actual stress distribution throughout the domain
            - Numbers show point ID and stress value at boundary points (MPa)
            """
        else:
            instructions = f"""
            **Simplified Analysis Results:**
            - Max shear stress: **{max_stress_mpa:.2f} MPa**
            - Analytical torsion formulas with interpolated stress field
            - Install DOLFINx for full FEA capability
            - Numbers show point ID and stress value (MPa)
            """
    else:
        instructions = f"""
        **Instructions:**
        - Currently editing: **{st.session_state.active_tab}**
        - Enter coordinates in the left panel using the text area
        - Minimum 3 points required for each shape
        - Points will be connected in order to form the shape
        - Use Quick Shapes or paste coordinate data
        - Analysis will use {'full DOLFINx FEA' if FEA_AVAILABLE else 'simplified analytical solver'}
        """
    
    st.pyplot(fig)
    st.info(instructions)

# Footer
st.markdown("---")
solver_status = "🚀 Full DOLFINx FEA" if FEA_AVAILABLE else "⚠️ Simplified Analytical"
st.markdown(f"""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>2D FEA Torsion Analysis</strong> - Web Version ({solver_status})</p>
    <p>Professional finite element analysis with GMSH meshing and DOLFINx solver</p>
    <p>Developed by <strong>Yacht Engineering Services</strong> | Built with Streamlit</p>
    <p>
        <a href='https://github.com/YachtEngineeringservices/2d-FEA' target='_blank'>GitHub Repository</a> | 
        <a href='https://github.com/YachtEngineeringservices/2d-FEA/releases' target='_blank'>Desktop Version</a>
    </p>
</div>
""", unsafe_allow_html=True)
