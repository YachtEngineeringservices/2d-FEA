"""
Streamlit Web Application for 2D FEA Torsion Analysis
This version uses the same DOLFINx solver and GMSH mesher as the desktop version
Professional FEA analysis in your web browser
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import sys
from fea import meshing, solver

# --- Page Config ---
# This must be the first Streamlit command in the script
st.set_page_config(
    page_title="2D FEA Torsion Analysis",
    page_icon=" torsional_analysis.png",
    layout="wide",
)

# --- Logging Setup ---
# Configure logging to display info level messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Check for DOLFINx ---
try:
    import dolfinx
    FEA_AVAILABLE = True
except ImportError:
    FEA_AVAILABLE = False

def run_full_fea(outer_points, inner_points, G, T, L, mesh_size=0.01):
    """
    Full FEA analysis using DOLFINx and GMSH (same as desktop version)
    No fallback - requires DOLFINx to be available
    """
    if not FEA_AVAILABLE:
        st.error("DOLFINx is not available in this environment. Cannot run Full FEA.")
        return None
        
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    st.info("🔧 Running full FEA analysis with DOLFINx...")
    
    try:
        # Generate mesh using GMSH (same as desktop)
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
        
        progress_bar.progress(25)
    
        # --- 2. Solve the FEA problem ---
        status_text.text("Solving with DOLFINx...")
        try:
            # Call the solver function from the solver module
            J, k, theta_deg, V_mag, tau_magnitude, max_stress_val = solver.solve_torsion(
                domain_mesh, facet_mesh, G, T, L
            )
            log.info(f"FEA solver completed: J={J:.4e}, k={k:.4e}, max_stress={max_stress_val/1e6:.2f} MPa")
        except Exception as e:
            log.error(f"FEA solver failed: {e}", exc_info=True)
            st.error(f"FEA analysis failed: {e}")
            st.error("This deployment requires full DOLFINx functionality")
            st.error("Please check the deployment logs and ensure DOLFINx is properly installed")
            return None
        
        progress_bar.progress(75)
    
        # --- 3. Extract results for visualization ---
        status_text.text("Extracting results...")
        
        # Extract stress field data for visualization BEFORE cleanup
        stress_field = None
        mesh_points = []
        mesh_triangles = None
        
        try:
            # Extract mesh coordinates and stress values directly from DOLFINx objects
            # This must be done BEFORE the solver cleans up temporary files
            if tau_magnitude is not None and V_mag is not None:
                # Get mesh coordinates from the function space
                mesh_coords = V_mag.tabulate_dof_coordinates()
                mesh_points = mesh_coords.copy()  # Make a copy to survive cleanup
                
                # Get stress values
                stress_field = tau_magnitude.x.array.copy()  # Make a copy to survive cleanup
                
                # Extract mesh triangulation topology
                domain = V_mag.mesh
                topology = domain.topology
                cell_map = topology.index_map(topology.dim)
                num_cells = cell_map.size_local
                
                # Get the triangles (cells) connectivity
                cells = topology.connectivity(topology.dim, 0)  # Cell-to-vertex connectivity
                mesh_triangles = np.zeros((num_cells, 3), dtype=np.int32)
                for i in range(num_cells):
                    cell_vertices = cells.links(i)
                    mesh_triangles[i] = cell_vertices[:3]  # Take first 3 vertices for triangle
                
                st.success("✅ Extracted full FEA mesh data with triangulation for visualization")
            else:
                st.warning("⚠️ Could not extract DOLFINx mesh data")
            
        except Exception as e:
            st.warning(f"Could not extract mesh visualization data: {e}")
            mesh_triangles = None
        
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
            'mesh_triangles': mesh_triangles,
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
if 'show_stress' not in st.session_state:
    st.session_state.show_stress = False
        
def visualize_stress_distribution(results, show_stress):
    """Generates and displays the stress distribution plot."""
    
    # --- Plotting Setup ---
    # Use a modern, clean style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get all defined points for framing the plot
    all_points = st.session_state.outer_points + st.session_state.inner_points
    
    if show_stress:
        # --- Full FEA Stress Visualization ---
        if (results and results.get('solver_type') == 'full_fea' and 
            results.get('stress_field') is not None and 
            results.get('mesh_points') is not None and
            results.get('mesh_triangles') is not None):
            
            try:
                # Extract data from results
                mesh_points = results['mesh_points']
                stress_field = results['stress_field']
                mesh_triangles = results['mesh_triangles']
                
                # Convert to display units (mm and MPa)
                mesh_x = mesh_points[:, 0] * 1000
                mesh_y = mesh_points[:, 1] * 1000
                stress_mpa = stress_field / 1e6

                # --- Create Triangulation (same as desktop) ---
                import matplotlib.tri as tri
                
                # Filter out any invalid triangles that might reference non-existent nodes
                max_node_idx = len(mesh_x) - 1
                valid_triangles = mesh_triangles[~np.any(mesh_triangles > max_node_idx, axis=1)]
                triang = tri.Triangulation(mesh_x, mesh_y, valid_triangles)
                
                # --- Masking Logic (same as desktop) ---
                # Calculate triangle centers to determine which are inside the geometry
                triangle_centers_x = triang.x[triang.triangles].mean(axis=1)
                triangle_centers_y = triang.y[triang.triangles].mean(axis=1)

                from matplotlib.path import Path
                
                # 1. Create mask for triangles OUTSIDE the outer boundary
                outer_array = np.array(st.session_state.outer_points)
                outer_path = Path(outer_array)
                final_mask = ~outer_path.contains_points(np.column_stack([triangle_centers_x, triangle_centers_y]))

                # 2. If there's a hole, add triangles INSIDE the hole to the mask
                if len(st.session_state.inner_points) >= 3:
                    inner_array = np.array(st.session_state.inner_points)
                    inner_path = Path(inner_array)
                    inside_hole_mask = inner_path.contains_points(np.column_stack([triangle_centers_x, triangle_centers_y]))
                    final_mask |= inside_hole_mask
                
                # Apply the final mask to the triangulation
                triang.set_mask(final_mask)
                
                # --- Plotting (same as desktop) ---
                # Use tripcolor for direct plotting of FEA mesh results
                contour = ax.tripcolor(triang, stress_mpa, shading='gouraud', cmap='jet')
                
                # Add contour lines for better definition
                ax.tricontour(triang, stress_mpa, levels=10, colors='black', alpha=0.3, linewidths=0.5)
                
                # Add colorbar
                cbar = fig.colorbar(contour, ax=ax, shrink=0.8, pad=0.02)
                cbar.set_label('Shear Stress (MPa)', rotation=270, labelpad=20, fontsize=14)
                
                # Add max stress annotation
                max_stress = stress_mpa.max()
                ax.text(0.02, 0.98, f"Max Stress: {max_stress:.1f} MPa", 
                       transform=ax.transAxes, fontsize=14, verticalalignment='top', 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

            except Exception as e:
                st.error(f"Visualization error: {e}")
                log.error(f"Failed to create FEA stress visualization: {e}", exc_info=True)

    # --- Plot Styling ---
    ax.set_title("Shear Stress Distribution", fontsize=16, pad=20)
    ax.set_xlabel("X (mm)", fontsize=14)
    ax.set_ylabel("Y (mm)", fontsize=14)
    ax.set_aspect('equal', adjustable='box')
    
    # Frame the plot using the geometry points
    if all_points:
        x_coords, y_coords = zip(*all_points)
        ax.set_xlim(min(x_coords) - 50, max(x_coords) + 50)
        ax.set_ylim(min(y_coords) - 50, max(y_coords) + 50)
    
    # Draw geometry outline if not showing stress
    if not show_stress and st.session_state.outer_points:
        outer_poly = plt.Polygon(st.session_state.outer_points, edgecolor='blue', fill=False, linewidth=1.5)
        ax.add_patch(outer_poly)
        if st.session_state.inner_points:
            inner_poly = plt.Polygon(st.session_state.inner_points, edgecolor='blue', fill=False, linewidth=1.5)
            ax.add_patch(inner_poly)

    # Display the plot in Streamlit
    st.pyplot(fig)

def main():
    # --- Sidebar ---
    with st.sidebar:
        st.header("Geometry Definition")
        
        # Tab selection for outer vs inner geometry
        st.session_state.active_tab = st.radio(
            "Edit Geometry:",
            ("Outer Shape", "Inner Hole"),
            horizontal=True,
        )
        
        # Point entry form
        with st.form(key='add_point_form'):
            st.write(f"Add points for **{st.session_state.active_tab}**")
            
            # Input for new points
            point_input = st.text_area(
                "Enter Points (x,y per line or single x,y):",
                height=100,
                placeholder="Examples:\\n10, 20\\n30 40\\n(50, 60)",
                key="point_input_area" # Add a key to persist the input
            )
            
            # Buttons for adding points
            add_points_button = st.form_submit_button(label="Add Points")
            replace_all_button = st.form_submit_button(label="Replace All")

            if add_points_button or replace_all_button:
                try:
                    new_points = []
                    # Use the correct key to get the value from session state
                    input_text = st.session_state.point_input_area
                    lines = input_text.strip().splitlines() # Use splitlines() for better handling of newlines
                    for line in lines:
                        line = line.strip().replace('(', '').replace(')', '')
                        if not line: continue # Skip empty lines
                        
                        if ',' in line:
                            parts = line.split(',')
                        else:
                            parts = line.split()
                        
                        if len(parts) == 2:
                            x, y = float(parts[0].strip()), float(parts[1].strip())
                            new_points.append([x, y])
                    
                    if new_points:
                        target_list = st.session_state.outer_points if st.session_state.active_tab == "Outer Shape" else st.session_state.inner_points
                        
                        if replace_all_button:
                            target_list.clear()
                            target_list.extend(new_points)
                        else:
                            target_list.extend(new_points)
                        
                        st.session_state.show_stress = False # Reset on geometry change
                        st.session_state.point_input_area = "" # Clear the input area
                        st.rerun()

                except ValueError:
                    st.error("Invalid input format. Please use x,y or x y.")

        # Display and manage current points
        current_points = st.session_state.outer_points if st.session_state.active_tab == "Outer Shape" else st.session_state.inner_points
        
        if current_points:
            df = pd.DataFrame(current_points, columns=['X (mm)', 'Y (mm)'])
            st.dataframe(df, height=200)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Undo Last", use_container_width=True):
                    if current_points:
                        current_points.pop()
                        st.rerun()
            with col2:
                if st.button("Clear All", use_container_width=True):
                    if current_points:
                        current_points.clear()
                        st.rerun()
        
        # Quick shapes
        st.write("---")
        st.write("**Quick Shapes**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Rectangle (100x50)", use_container_width=True):
                st.session_state.outer_points = [[-50,-25], [50,-25], [50,25], [-50,25]]
                st.session_state.inner_points = []
                st.rerun()
            if st.button("Triangle (Isosceles)", use_container_width=True):
                st.session_state.outer_points = [[-50,0], [50,0], [0,86.6]]
                st.session_state.inner_points = []
                st.rerun()
        with col2:
            if st.button("Circle (r=50)", use_container_width=True):
                st.session_state.outer_points = [[50*np.cos(a), 50*np.sin(a)] for a in np.linspace(0, 2*np.pi, 30)]
                st.session_state.inner_points = []
                st.rerun()
            if st.button("L-Shape (100x100)", use_container_width=True):
                st.session_state.outer_points = [[0,0], [100,0], [100,20], [20,20], [20,100], [0,100]]
                st.session_state.inner_points = []
                st.rerun()

        # --- Analysis Controls ---
        st.header("Analysis Controls")
    
        # Inputs for material properties and loading
        shear_modulus = st.number_input("Shear Modulus (G) [MPa]:", value=80e3, format="%g")
        applied_torque = st.number_input("Applied Torque (T) [N-m]:", value=1000.0, format="%f")
        beam_length = st.number_input("Beam Length (L) [m]:", value=2.0, format="%f")
            
        # Mesh size input for FEA (now always shown)
        mesh_size = st.number_input("Mesh Size (mm):", value=5.0, min_value=0.1, max_value=50.0, format="%.1f")
        
        # Generate & Solve button
        if st.button("Generate Mesh & Solve", type="primary", use_container_width=True):
            if len(st.session_state.outer_points) < 3:
                st.error("Please define at least 3 points for the outer shape.")
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
                    
                    if results:
                        st.session_state.results = results
                        st.session_state.show_stress = True
                    else:
                        st.error("Analysis failed. Check logs for details.")
                        st.session_state.results = None
                        st.session_state.show_stress = False
                    
                    st.rerun()

    # --- Main Content ---
    st.title("2D FEA Torsional Analysis")
    
    # --- Results Display ---
    if st.session_state.results:
        st.header("Results")
        results = st.session_state.results
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Torsional Constant (J)", f"{results['polar_moment']:.4e} mm⁴")
        col2.metric("Torsional Stiffness (k)", f"{results['stiffness']:.4e} Nm/rad")
        col3.metric("Angle of Twist (θ)", f"{results['twist_angle']:.4f} rad")
        col4.metric("Max Shear Stress (τ_max)", f"{results['max_shear_stress'] / 1e6:.2f} MPa")
        
        # --- Visualization ---
        st.header("Stress Distribution")
        
        # View toggle
        view_mode = st.radio("View:", ("Geometry", "Stress Results"), horizontal=True)
        
        if view_mode == "Stress Results":
            st.session_state.show_stress = True
        else:
            st.session_state.show_stress = False
            
        visualize_stress_distribution(st.session_state.results, st.session_state.show_stress)
    else:
        # Show a placeholder or instructions if no results yet
        st.info("Define a geometry and run the analysis to see the results.")
        visualize_stress_distribution(None, False)

if __name__ == "__main__":
    # --- Initialize Session State ---
    if 'outer_points' not in st.session_state:
        st.session_state.outer_points = []
    if 'inner_points' not in st.session_state:
        st.session_state.inner_points = []
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Outer Shape"
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'show_stress' not in st.session_state:
        st.session_state.show_stress = False
        
    main()
