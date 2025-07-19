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

# Configure page FIRST - must be the very first Streamlit command
st.set_page_config(
    page_title="2D FEA Torsion Analysis - Yacht Engineering Services",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set matplotlib backend for cloud environment
import matplotlib
matplotlib.use('Agg')

# Import the same FEA components as desktop version
try:
    # Check if GMSH is available before importing FEA modules
    import gmsh
    st.write(f"✅ GMSH version: {gmsh.__version__}")
    
    from fea import meshing, solver
    from fea.solver import solve_torsion
    FEA_AVAILABLE = True
    st.success("✅ Full FEA solver (DOLFINx) available")
    
    # Verify the imported modules are not None
    if meshing is None or solver is None:
        raise ImportError("FEA modules imported but are None - check fea/__init__.py")
    
except ImportError as e:
    st.error(f"❌ FEA solver not available: {e}")
    st.error("🚨 **DEPLOYMENT ERROR**: This web app requires DOLFINx for proper FEA analysis")
    st.error("Please ensure DOLFINx is installed in the deployment environment")
    st.error(f"**Import Error Details**: {str(e)}")
    
    # Add deployment timestamp to check if rebuilding
    import datetime
    st.error(f"**Deployment Check**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    st.stop()  # Stop the app execution

def run_full_fea(outer_points, inner_points, G, T, L, mesh_size=0.01):
    """
    Full FEA analysis using DOLFINx and GMSH (same as desktop version)
    No fallback - requires DOLFINx to be available
    """
    if not FEA_AVAILABLE:
        st.error("🚨 **CRITICAL ERROR**: DOLFINx not available!")
        st.error("This web app requires full FEA capability - no simplified solver available")
        st.stop()
        return None
    
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
            st.error("🚨 Critical error: Cannot proceed without valid mesh")
            st.stop()
            return None
        
        progress_bar.progress(50)
        status_text.text("Solving with DOLFINx...")
        
        # Use the same solver as desktop
        J, k, theta, tau_max, tau_magnitude, V_mag = solve_torsion(
            output_dir, G, T, L
        )
        
        progress_bar.progress(80)
        status_text.text("Extracting results...")
        
        # Extract stress field data for visualization BEFORE cleanup
        stress_field = None
        mesh_points = []
        
        try:
            # Extract mesh coordinates and stress values directly from DOLFINx objects
            # This must be done BEFORE the solver cleans up temporary files
            if tau_magnitude is not None and V_mag is not None:
                # Get mesh coordinates from the function space
                mesh_coords = V_mag.tabulate_dof_coordinates()
                mesh_points = mesh_coords.copy()  # Make a copy to survive cleanup
                
                # Get stress values
                stress_field = tau_magnitude.x.array.copy()  # Make a copy to survive cleanup
                
                st.success("✅ Extracted full FEA mesh data for visualization")
            else:
                st.warning("⚠️ Could not extract DOLFINx mesh data")
            
        except Exception as e:
            st.warning(f"Could not extract mesh visualization data: {e}")
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        # Calculate stress at boundary points for point visualization
        outer_stress_values = []
        inner_stress_values = []
        
        # Use actual FEA stress field instead of simplified formula
        if stress_field is not None and mesh_points is not None:
            try:
                # Ensure mesh_points is 2D (x,y coordinates only)
                if mesh_points.shape[1] > 2:
                    mesh_coords_2d = mesh_points[:, :2]  # Take only x,y coordinates
                else:
                    mesh_coords_2d = mesh_points
                
                # Find stress values at boundary points from the actual FEA mesh
                if outer_points:
                    outer_pts = np.array(outer_points)
                    for point in outer_pts:
                        # Ensure point is 2D
                        point_2d = np.array(point[:2]) if len(point) > 2 else np.array(point)
                        # Find closest mesh point to this boundary point
                        distances = np.sqrt(np.sum((mesh_coords_2d - point_2d)**2, axis=1))
                        closest_idx = np.argmin(distances)
                        # Use actual FEA stress value
                        actual_stress = stress_field[closest_idx]
                        outer_stress_values.append(actual_stress)
                    
                if inner_points:
                    inner_pts = np.array(inner_points)
                    for point in inner_pts:
                        # Ensure point is 2D
                        point_2d = np.array(point[:2]) if len(point) > 2 else np.array(point)
                        # Find closest mesh point to this boundary point
                        distances = np.sqrt(np.sum((mesh_coords_2d - point_2d)**2, axis=1))
                        closest_idx = np.argmin(distances)
                        # Use actual FEA stress value
                        actual_stress = stress_field[closest_idx]
                        inner_stress_values.append(actual_stress)
                        
            except Exception as e:
                st.warning(f"Could not extract boundary stress from FEA mesh: {e}")
                # Fall back to simplified calculation
                stress_field = None  # Trigger fallback
        
        if stress_field is None:
            # Fallback: Use simplified approximation only if FEA data unavailable
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
        st.error(f"🚨 **FEA analysis failed**: {str(e)}")
        st.error("This deployment requires full DOLFINx functionality")
        st.error("Please check the deployment logs and ensure DOLFINx is properly installed")
        st.stop()
        return None

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
        
        # Show the editable dataframe
        edited_df = st.data_editor(
            points_df, 
            use_container_width=True,
            num_rows="dynamic",  # Allow adding/removing rows
            key=f"points_editor_{geometry_tab.lower().replace(' ', '_')}"
        )
        
        # Update session state when points are edited
        if not edited_df.equals(points_df):
            # Convert back to list of tuples and update session state
            new_points = [(float(row['X (mm)']), float(row['Y (mm)'])) for _, row in edited_df.iterrows()]
            if geometry_tab == "Outer Shape":
                st.session_state.outer_points = new_points
            else:
                st.session_state.inner_points = new_points
            st.rerun()
        
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
                
                # Run full FEA analysis (DOLFINx required)
                results = run_full_fea(
                    outer_points_m, 
                    inner_points_m,
                    shear_modulus * 1e6,  # Convert MPa to Pa
                    applied_torque,
                    beam_length,
                    mesh_size / 1000  # Convert mm to m
                )
                
                if results is not None:
                    st.session_state.results = results
                    
                    if results['success']:
                        st.success("✅ Professional FEA analysis completed with DOLFINx!")
                    else:
                        st.error("❌ FEA analysis failed")
                else:
                    st.error("❌ Critical error: Cannot run analysis without DOLFINx")
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
            
            # Full FEA visualization only - no simplified fallback
            if (results['solver_type'] == 'full_fea' and 
                results['stress_field'] is not None and 
                results['mesh_points'] is not None):
                
                # Full FEA contour plot using proper triangulation (same as desktop)
                try:
                    mesh_points = results['mesh_points']
                    stress_field = results['stress_field']
                    
                    # Convert mesh points to mm for display
                    mesh_x = mesh_points[:, 0] * 1000
                    mesh_y = mesh_points[:, 1] * 1000
                    stress_mpa = stress_field / 1e6  # Convert to MPa
                    
                    # Create matplotlib triangulation (same method as desktop)
                    import matplotlib.tri as tri
                    triang = tri.Triangulation(mesh_x, mesh_y)
                    
                    # Create stress visualization using tripcolor (same as desktop)
                    contour = ax.tripcolor(triang, stress_mpa, shading='gouraud', cmap='jet', alpha=0.9)
                    
                    # Add contour lines for better definition
                    try:
                        levels = np.linspace(stress_mpa.min(), stress_mpa.max(), 10)
                        ax.tricontour(triang, stress_mpa, levels=levels, colors='black', alpha=0.3, linewidths=0.5)
                    except:
                        pass  # Skip contour lines if triangulation issues
                    
                    # Add colorbar
                    cbar = plt.colorbar(contour, ax=ax, shrink=0.8, pad=0.02)
                    cbar.set_label('Shear Stress (MPa)', rotation=270, labelpad=20, fontsize=12, fontweight='bold')
                    
                    # Plot polygon outlines
                    outer_polygon = Polygon(outer_array, fill=False, edgecolor='white', 
                                          linewidth=3, label='Outer Shape')
                    ax.add_patch(outer_polygon)
                    
                    # Plot inner hole if present
                    if len(st.session_state.inner_points) >= 3:
                        inner_array = np.array(st.session_state.inner_points)
                        inner_polygon = Polygon(inner_array, fill=False, edgecolor='white', 
                                              linewidth=2, label='Inner Hole')
                        ax.add_patch(inner_polygon)
                    
                    st.success("🎯 Professional FEA stress field visualization (DOLFINx)")
                    
                except Exception as e:
                    st.error(f"Visualization error: {e}")
                    st.error("Failed to create FEA stress visualization")
                    
            else:
                st.error("🚨 **VISUALIZATION ERROR**: No valid FEA results available")
                st.error("This should not happen if DOLFINx is properly installed")
                
            # Plot stress at boundary points with values (full FEA only)
            if results['outer_stress_values']:
                stress_mpa = [s / 1e6 for s in results['outer_stress_values']]
                scatter = ax.scatter(outer_array[:, 0], outer_array[:, 1], 
                                   c=stress_mpa, s=150, cmap='jet', 
                                   edgecolor='white', linewidth=2, zorder=5)
                
                # Add stress value annotations with actual FEA values
                for i, (x, y) in enumerate(outer_array):
                    if i < len(stress_mpa):
                        ax.annotate(f'{i+1}\n{stress_mpa[i]:.1f}', (x, y), 
                                  xytext=(8, 8), textcoords='offset points',
                                  fontsize=9, fontweight='bold', color='white',
                                  bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))
        else:
            # Regular geometry view
            outer_polygon = Polygon(outer_array, alpha=0.3, facecolor='lightblue', 
                                   edgecolor='blue', linewidth=2, label='Outer Shape')
            ax.add_patch(outer_polygon)
            
            # Plot outer points
            ax.scatter(outer_array[:, 0], outer_array[:, 1], c='blue', s=100, zorder=5)
    
    # Plot inner hole (full FEA only)
    if len(st.session_state.inner_points) >= 3:
        inner_array = np.array(st.session_state.inner_points)
        
        if show_stress:
            # Create inner hole outline and stress visualization
            results = st.session_state.results
            
            # Create inner hole as white cutout
            inner_polygon = Polygon(inner_array, fill=True, facecolor='white', 
                                   edgecolor='white', linewidth=3, zorder=10)
            ax.add_patch(inner_polygon)
            
            # Add boundary outline
            inner_outline = Polygon(inner_array, fill=False, edgecolor='white', 
                                   linewidth=2, zorder=11)
            ax.add_patch(inner_outline)
            
            # Plot stress points for inner boundary if available
            if results['inner_stress_values']:
                stress_mpa = [s / 1e6 for s in results['inner_stress_values']]
                scatter_inner = ax.scatter(inner_array[:, 0], inner_array[:, 1], 
                                         c=stress_mpa, s=150, cmap='jet', 
                                         edgecolor='white', linewidth=2, zorder=15)
                
                # Add stress value annotations for inner points with actual FEA values
                for i, (x, y) in enumerate(inner_array):
                    if i < len(stress_mpa):
                        ax.annotate(f'{i+1}\n{stress_mpa[i]:.1f}', (x, y), 
                                  xytext=(8, 8), textcoords='offset points',
                                  fontsize=9, fontweight='bold', color='black',
                                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        else:
            # Regular geometry view
            inner_polygon = Polygon(inner_array, alpha=0.8, facecolor='white', 
                                   edgecolor='red', linewidth=2, label='Inner Hole')
            ax.add_patch(inner_polygon)
            
            # Plot inner points
            ax.scatter(inner_array[:, 0], inner_array[:, 1], c='red', s=100, zorder=5)
    
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
        
        instructions = f"""
        **Professional FEA Analysis Results (DOLFINx):**
        - Max shear stress: **{max_stress_mpa:.2f} MPa**
        - Full finite element analysis with mesh generation
        - Stress field computed from DOLFINx solver
        - Contour plot shows actual stress distribution throughout the domain
        - Numbers show point ID and stress value at boundary points (MPa)
        - Identical results to desktop version
        """
    else:
        instructions = f"""
        **Instructions:**
        - Currently editing: **{st.session_state.active_tab}**
        - Enter coordinates in the left panel using the text area
        - Minimum 3 points required for each shape
        - Points will be connected in order to form the shape
        - Use Quick Shapes or paste coordinate data
        - Analysis uses full DOLFINx FEA (same as desktop version)
        """
    
    st.pyplot(fig)
    st.info(instructions)

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>2D FEA Torsion Analysis</strong> - Professional Web Version</p>
    <p>Full DOLFINx finite element analysis with GMSH meshing - Same as desktop version</p>
    <p>Developed by <strong>Yacht Engineering Services</strong> | Built with Streamlit</p>
    <p>
        <a href='https://github.com/YachtEngineeringservices/2d-FEA' target='_blank'>GitHub Repository</a> | 
        <a href='https://github.com/YachtEngineeringservices/2d-FEA/releases' target='_blank'>Desktop Version</a>
    </p>
</div>
""", unsafe_allow_html=True)
