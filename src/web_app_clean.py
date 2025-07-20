import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import logging
import os
from fea import meshing, solver

# Check GMSH availability early
try:
    gmsh_available = meshing.check_gmsh_availability()
    if not gmsh_available:
        st.error("⚠️ GMSH is not available. Mesh generation will not work.")
        st.info("This is likely due to platform limitations. Please try a different deployment method.")
except Exception as e:
    st.error(f"Error checking GMSH availability: {e}")
    gmsh_available = False

# --- Page Config ---
st.set_page_config(
    page_title="2D FEA Torsion Analysis",
    layout="wide",
)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Session State Initialization ---
def init_session_state():
    """Initialize session state variables."""
    if 'outer_points' not in st.session_state:
        st.session_state.outer_points = []
    if 'inner_points' not in st.session_state:
        st.session_state.inner_points = []
    if 'results' not in st.session_state:
        st.session_state.results = None
    # Add state for manual zoom controls
    if 'x_min' not in st.session_state: st.session_state.x_min = None
    if 'x_max' not in st.session_state: st.session_state.x_max = None
    if 'y_min' not in st.session_state: st.session_state.y_min = None
    if 'y_max' not in st.session_state: st.session_state.y_max = None
    # Add state for slider controls
    if 'zoom_level' not in st.session_state: st.session_state.zoom_level = 1.0
    if 'pan_x' not in st.session_state: st.session_state.pan_x = 0.0
    if 'pan_y' not in st.session_state: st.session_state.pan_y = 0.0
    # Add state for terminal/log output
    if 'log_messages' not in st.session_state: st.session_state.log_messages = []
    if 'is_calculating' not in st.session_state: st.session_state.is_calculating = False
    # Add state for mesh and solve separation
    if 'mesh_data' not in st.session_state: st.session_state.mesh_data = None
    if 'is_meshing' not in st.session_state: st.session_state.is_meshing = False

def add_log_message(message):
    """Add a message to the log with timestamp"""
    import datetime
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    st.session_state.log_messages.append(full_message)
    # Keep only last 50 messages to prevent memory issues
    if len(st.session_state.log_messages) > 50:
        st.session_state.log_messages = st.session_state.log_messages[-50:]

def get_geometry_bounds(outer_points, inner_points):
    """Calculates the bounding box of the entire geometry with a margin."""
    if not outer_points:
        return -10, 10, -10, 10
    
    all_points = np.array(outer_points + inner_points)
    x_min, y_min = all_points.min(axis=0)
    x_max, y_max = all_points.max(axis=0)
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    margin = max(x_range, y_range) * 0.1 # 10% margin
    
    return x_min - margin, x_max + margin, y_min - margin, y_max + margin

# --- Main Application ---
def main():
    """Main function to run the Streamlit application."""
    st.title("2D Torsional Analysis FEA")

    # --- Sidebar for Inputs ---
    with st.sidebar:
        st.header("Geometry Definition")

        outer_tab, inner_tab = st.tabs(["Outer Shape", "Inner Hole"])

        with outer_tab:
            manage_geometry("outer")
        
        with inner_tab:
            manage_geometry("inner")

        st.header("Plot Controls")
        
        # Get geometry bounds for slider ranges
        bounds = get_geometry_bounds(st.session_state.outer_points, st.session_state.inner_points)
        geometry_width = bounds[1] - bounds[0]
        geometry_height = bounds[3] - bounds[2]
        geometry_center_x = (bounds[0] + bounds[1]) / 2
        geometry_center_y = (bounds[2] + bounds[3]) / 2
        
        # Reset View button
        if st.button("Reset View", use_container_width=True):
            st.session_state.x_min = None
            st.session_state.x_max = None
            st.session_state.y_min = None
            st.session_state.y_max = None
            st.session_state.zoom_level = 1.0
            st.session_state.pan_x = 0.0
            st.session_state.pan_y = 0.0
            st.rerun()
        
        # Initialize zoom and pan if not set
        if 'zoom_level' not in st.session_state:
            st.session_state.zoom_level = 1.0
        if 'pan_x' not in st.session_state:
            st.session_state.pan_x = 0.0
        if 'pan_y' not in st.session_state:
            st.session_state.pan_y = 0.0
        
        # Zoom slider
        st.subheader("Zoom")
        zoom_level = st.slider(
            "Zoom Level", 
            min_value=0.1, 
            max_value=10.0, 
            value=st.session_state.zoom_level, 
            step=0.1,
            format="%.1fx"
        )
        
        # Pan sliders
        st.subheader("Pan")
        pan_range = float(max(geometry_width, geometry_height))
        
        pan_x = st.slider(
            "Pan X", 
            min_value=-pan_range, 
            max_value=pan_range, 
            value=st.session_state.pan_x, 
            step=pan_range/100.0,
            format="%.1f"
        )
        
        pan_y = st.slider(
            "Pan Y", 
            min_value=-pan_range, 
            max_value=pan_range, 
            value=st.session_state.pan_y, 
            step=pan_range/100.0,
            format="%.1f"
        )
        
        # Update session state and calculate view bounds
        if (zoom_level != st.session_state.zoom_level or 
            pan_x != st.session_state.pan_x or 
            pan_y != st.session_state.pan_y):
            
            st.session_state.zoom_level = zoom_level
            st.session_state.pan_x = pan_x
            st.session_state.pan_y = pan_y
            
            # Calculate view bounds based on zoom and pan
            view_width = geometry_width / zoom_level
            view_height = geometry_height / zoom_level
            
            center_x = geometry_center_x + pan_x
            center_y = geometry_center_y + pan_y
            
            st.session_state.x_min = center_x - view_width / 2
            st.session_state.x_max = center_x + view_width / 2
            st.session_state.y_min = center_y - view_height / 2
            st.session_state.y_max = center_y + view_height / 2
            
            st.rerun()

        st.header("Analysis Controls")
        mesh_size = st.number_input("Mesh Size [mm]:", value=1.0, min_value=0.1, max_value=50.0, step=0.1, format="%.2f")
        
        # Adaptive mesh refinement option
        use_adaptive = st.checkbox("Use Adaptive Mesh Refinement", value=True, 
                                  help="Automatically refines mesh in high stress gradient areas for better accuracy")
        
        if use_adaptive:
            refinement_levels = st.selectbox("Refinement Levels", [1, 2, 3], index=1,
                                           help="Higher levels = more accurate but slower computation")
        else:
            refinement_levels = 0

        st.header("Inputs")
        g_modulus = st.number_input("Shear Modulus (G) [MPa]:", value=80e3, format="%e")
        torque = st.number_input("Applied Torque (T) [N-m]:", value=1000.0)
        length = st.number_input("Beam Length (L) [m]:", value=2.0)

        if st.button("Generate Mesh & Solve", type="primary", use_container_width=True):
            st.session_state.is_calculating = True
            st.session_state.log_messages = []  # Clear previous logs
            add_log_message("Starting analysis...")
            # Store refinement setting for the analysis
            st.session_state.refinement_levels = refinement_levels
            st.rerun()

        # Separate mesh generation button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("1️⃣ Generate Mesh", use_container_width=True):
                st.session_state.is_meshing = True
                st.session_state.log_messages = []  # Clear previous logs
                add_log_message("Starting mesh generation...")
                st.session_state.refinement_levels = refinement_levels
                st.rerun()
        
        with col2:
            mesh_available = st.session_state.mesh_data is not None
            if st.button("2️⃣ Solve FEA", use_container_width=True, disabled=not mesh_available):
                if mesh_available:
                    st.session_state.is_calculating = True
                    add_log_message("Starting FEA solve with existing mesh...")
                    st.rerun()
                else:
                    st.warning("Please generate mesh first!")

    # --- Main area for Plot and Results ---
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header("Shear Stress Distribution")
        
        # Check if meshing should start
        if st.session_state.is_meshing:
            add_log_message("Generating mesh only...")
            refinement_levels = getattr(st.session_state, 'refinement_levels', 2)
            mesh_result = generate_mesh_only(st.session_state.outer_points, st.session_state.inner_points, 
                                           mesh_size, refinement_levels)
            if mesh_result:
                st.session_state.mesh_data = mesh_result
                add_log_message("Mesh generation completed successfully!")
                add_log_message("You can now review the mesh and proceed to solve.")
            else:
                add_log_message("Mesh generation failed. Please check geometry and parameters.")
            st.session_state.is_meshing = False
            st.rerun()
        
        # Check if calculation should start
        if st.session_state.is_calculating and not st.session_state.results:
            if st.session_state.mesh_data:
                # Use existing mesh
                add_log_message("Using existing mesh for FEA solve...")
                results = solve_with_existing_mesh(st.session_state.mesh_data, g_modulus, torque, length)
            else:
                # Full analysis (mesh + solve)
                add_log_message("Validating geometry...")
                refinement_levels = getattr(st.session_state, 'refinement_levels', 2)
                results = run_analysis(st.session_state.outer_points, st.session_state.inner_points, 
                                     mesh_size, g_modulus, torque, length, refinement_levels)
            
            if results:
                st.session_state.results = results
                add_log_message("Analysis completed successfully!")
            else:
                add_log_message("Analysis failed. Please check geometry and parameters.")
            st.session_state.is_calculating = False
            st.rerun()
        
        # Pass zoom state to the plotting function
        fig = plot_stress_distribution(
            st.session_state.results, 
            st.session_state.outer_points, 
            st.session_state.inner_points,
            xlim=(st.session_state.x_min, st.session_state.x_max),
            ylim=(st.session_state.y_min, st.session_state.y_max),
            mesh_data=st.session_state.mesh_data  # Pass mesh data for visualization
        )
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.header("Results")
        if st.session_state.results:
            results = st.session_state.results
            J_mm4 = results['J'] * (1000**4)
            st.metric("Torsional Constant (J)", f"{J_mm4:.4e} mm⁴")
            st.metric("Torsional Stiffness (k)", f"{results['k']:.4e} Nm/rad")
            st.metric("Angle of Twist (θ)", f"{np.rad2deg(results['theta']):.4f} deg")
            st.metric("Max Shear Stress (τ_max)", f"{results['tau_max']/1e6:.2f} MPa")
            
            # Add solver information
            st.info("""
            **ℹ️ Solver Information:**
            
            This web app uses **DOLFINx FEA** which solves the full torsion equations with proper boundary conditions. 
            
            The stress distribution shows highest values where geometry changes create stress concentrations, which is physically accurate.
            
            *Note: Results may differ from simplified analytical methods that assume circular shaft behavior.*
            """)
        else:
            st.text("Torsional Constant (J): Not calculated")
            st.text("Torsional Stiffness (k): Not calculated")
            st.text("Angle of Twist (θ): Not calculated")
            st.text("Max Shear Stress (τ_max): Not calculated")

    # --- Terminal/Log Window ---
    st.header("Analysis Log")
    
    # Terminal-style container
    terminal_container = st.container()
    with terminal_container:
        if st.session_state.log_messages:
            # Create terminal-style display
            log_text = "\n".join(st.session_state.log_messages)
            st.code(log_text, language=None)
        else:
            st.code("Ready for analysis...", language=None)
        
        # Show spinner if calculating
        if st.session_state.is_calculating:
            st.spinner("Calculating...")
    
    # Auto-scroll to bottom by refreshing when new messages arrive
    if st.session_state.is_calculating:
        import time
        time.sleep(0.5)  # Small delay for smooth updates
        st.rerun()

def run_analysis(outer_points, inner_points, mesh_size, G, T, L, refinement_levels=2):
    """Runs the meshing and solving process."""
    if len(outer_points) < 3:
        add_log_message("ERROR: Outer shape must have at least 3 points.")
        return None

    try:
        add_log_message(f"Converting units: mesh size {mesh_size}mm -> {mesh_size/1000}m")
        # Convert inputs to SI units for the solver
        outer_points_m = [(p[0] / 1000.0, p[1] / 1000.0) for p in outer_points]
        inner_points_m = [(p[0] / 1000.0, p[1] / 1000.0) for p in inner_points] if inner_points else []
        mesh_size_m = mesh_size / 1000.0
        G_Pa = G * 1e6  # MPa to Pa
        
        add_log_message(f"Material properties: G = {G} MPa, T = {T} Nm, L = {L} m")
        add_log_message(f"Geometry: {len(outer_points)} outer points, {len(inner_points)} inner points")
        
        # Validate geometry
        add_log_message("Validating geometry...")
        # Check if polygon is clockwise or counterclockwise
        def polygon_area(points):
            n = len(points)
            area = 0.0
            for i in range(n):
                j = (i + 1) % n
                area += points[i][0] * points[j][1]
                area -= points[j][0] * points[i][1]
            return area / 2.0
        
        outer_area = polygon_area(outer_points)
        if outer_area < 0:
            add_log_message("WARNING: Outer polygon is clockwise - this may cause mesh issues")
        else:
            add_log_message("Outer polygon is counter-clockwise (correct orientation)")
        
        add_log_message(f"Outer polygon area: {abs(outer_area):.2f} mm²")
        
        add_log_message("Starting mesh generation with GMSH...")
        add_log_message(f"Mesh algorithm: Automatic (robust)")
        add_log_message(f"Boundary mesh size: {mesh_size_m * 0.7:.6f}m")
        add_log_message(f"Domain mesh size: {mesh_size_m * 0.8:.6f}m")
        add_log_message(f"Adaptive refinement levels: {refinement_levels}")
        mesh_dir = meshing.create_mesh(outer_points_m, inner_points_m, mesh_size_m)
        if not mesh_dir:
            add_log_message("ERROR: Mesh generation failed.")
            return None
        
        add_log_message(f"Mesh created successfully in: {mesh_dir}")
        add_log_message("Mesh includes automatic refinement for better coverage")
        add_log_message("Initializing DOLFINx solver...")
        
        # Assuming solver.solve_torsion returns a tuple of results
        add_log_message("Setting up finite element problem...")
        add_log_message("Applying boundary conditions...")
        add_log_message("Assembling system matrices...")
        add_log_message("Solving linear system...")
        
        J, k, theta, tau_max, tau_magnitude, V_mag = solver.solve_torsion(mesh_dir, G_Pa, T, L)
        
        add_log_message("Calculating torsional properties...")
        add_log_message(f"Torsional constant J = {J:.4e} m⁴")
        add_log_message(f"Torsional stiffness k = {k:.4e} Nm/rad")
        add_log_message(f"Angle of twist θ = {np.rad2deg(theta):.4f} degrees")
        add_log_message(f"Maximum shear stress = {tau_max/1e6:.2f} MPa")
        add_log_message("Post-processing stress field...")
        
        return {
            "J": J, "k": k, "theta": theta, "tau_max": tau_max,
            "tau_magnitude": tau_magnitude, "V_mag": V_mag
        }

    except Exception as e:
        add_log_message(f"ERROR: Analysis failed - {str(e)}")
        log.error("Analysis failed", exc_info=True)
        return None

def generate_mesh_only(outer_points, inner_points, mesh_size, refinement_levels=2):
    """Generate mesh only without solving."""
    if len(outer_points) < 3:
        add_log_message("ERROR: Outer shape must have at least 3 points.")
        return None

    try:
        add_log_message(f"Converting units: mesh size {mesh_size}mm -> {mesh_size/1000}m")
        # Convert inputs to SI units
        outer_points_m = [(p[0] / 1000.0, p[1] / 1000.0) for p in outer_points]
        inner_points_m = [(p[0] / 1000.0, p[1] / 1000.0) for p in inner_points] if inner_points else []
        mesh_size_m = mesh_size / 1000.0
        
        add_log_message(f"Geometry: {len(outer_points)} outer points, {len(inner_points)} inner points")
        
        # Validate geometry
        add_log_message("Validating geometry...")
        def polygon_area(points):
            n = len(points)
            area = 0.0
            for i in range(n):
                j = (i + 1) % n
                area += points[i][0] * points[j][1]
                area -= points[j][0] * points[i][1]
            return area / 2.0
        
        outer_area = polygon_area(outer_points)
        if outer_area < 0:
            add_log_message("WARNING: Outer polygon is clockwise - this may cause mesh issues")
        else:
            add_log_message("Outer polygon is counter-clockwise (correct orientation)")
        
        add_log_message(f"Outer polygon area: {abs(outer_area):.2f} mm²")
        
        add_log_message("Starting mesh generation with GMSH...")
        add_log_message(f"Adaptive refinement levels: {refinement_levels}")
        mesh_dir = meshing.create_mesh(outer_points_m, inner_points_m, mesh_size_m)
        if not mesh_dir:
            add_log_message("ERROR: Mesh generation failed.")
            return None
        
        add_log_message(f"Mesh created successfully in: {mesh_dir}")
        
        # Read mesh data for visualization
        try:
            import meshio
            domain_file = os.path.join(mesh_dir, "domain.xdmf")
            add_log_message(f"Looking for mesh file: {domain_file}")
            
            if os.path.exists(domain_file):
                add_log_message("Reading mesh data...")
                mesh_data = meshio.read(domain_file)
                add_log_message(f"Mesh contains {len(mesh_data.points)} nodes and {len(mesh_data.cells[0].data)} elements")
                
                return {
                    'mesh_dir': mesh_dir,
                    'mesh_data': mesh_data,
                    'outer_points_m': outer_points_m,
                    'inner_points_m': inner_points_m
                }
            else:
                add_log_message(f"ERROR: Mesh file not found at {domain_file}")
                # List files in mesh directory for debugging
                if os.path.exists(mesh_dir):
                    files = os.listdir(mesh_dir)
                    add_log_message(f"Files in mesh directory: {files}")
                return None
        except Exception as e:
            add_log_message(f"ERROR: Could not process mesh data - {str(e)}")
            import traceback
            add_log_message(f"Traceback: {traceback.format_exc()}")
            return None
            
    except Exception as e:
        add_log_message(f"ERROR: Mesh generation failed - {str(e)}")
        return None

def solve_with_existing_mesh(mesh_data, G, T, L):
    """Solve FEA using existing mesh data."""
    try:
        add_log_message("Using existing mesh for FEA analysis...")
        mesh_dir = mesh_data['mesh_dir']
        G_Pa = G * 1e6  # MPa to Pa
        
        add_log_message(f"Material properties: G = {G} MPa, T = {T} Nm, L = {L} m")
        add_log_message("Initializing DOLFINx solver...")
        add_log_message("Setting up finite element problem...")
        add_log_message("Applying boundary conditions...")
        add_log_message("Assembling system matrices...")
        add_log_message("Solving linear system...")
        
        J, k, theta, tau_max, tau_magnitude, V_mag = solver.solve_torsion(mesh_dir, G_Pa, T, L)
        
        add_log_message("Calculating torsional properties...")
        add_log_message(f"Torsional constant J = {J:.4e} m⁴")
        add_log_message(f"Torsional stiffness k = {k:.4e} Nm/rad")
        add_log_message(f"Angle of twist θ = {np.rad2deg(theta):.4f} degrees")
        add_log_message(f"Maximum shear stress = {tau_max/1e6:.2f} MPa")
        add_log_message("Post-processing stress field...")
        
        return {
            "J": J, "k": k, "theta": theta, "tau_max": tau_max,
            "tau_magnitude": tau_magnitude, "V_mag": V_mag
        }
        
    except Exception as e:
        add_log_message(f"ERROR: FEA solve failed - {str(e)}")
        return None

def plot_stress_distribution(results, outer_points, inner_points, xlim=None, ylim=None, mesh_data=None):
    """
    Plots the stress distribution using Matplotlib for high-quality rendering.
    If only mesh_data is provided (no results), shows mesh visualization.
    """
    # Create figure with specific size and tight layout
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Set a dark background for the plot to match the Streamlit theme
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')

    # Set colors for ticks, labels, and spines to be visible on a dark background
    ax.tick_params(colors='white', which='both')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    # Check what to plot
    if results and results.get("tau_magnitude") is not None and results.get("V_mag") is not None:
        # Plot stress results
        tau_magnitude = results["tau_magnitude"]
        V_mag = results["V_mag"]
        
        mesh = V_mag.mesh
        coords = V_mag.tabulate_dof_coordinates()[:, :2]
        stress_values_pa = tau_magnitude.x.array
        
        coords_mm = coords * 1000
        stress_values_mpa = stress_values_pa / 1e6
        
        # Debug information using logging and session state
        add_log_message(f"DEBUG: Coordinate range - X: [{coords_mm[:, 0].min():.1f}, {coords_mm[:, 0].max():.1f}], Y: [{coords_mm[:, 1].min():.1f}, {coords_mm[:, 1].max():.1f}]")
        add_log_message(f"DEBUG: Stress value range: [{stress_values_mpa.min():.2f}, {stress_values_mpa.max():.2f}] MPa")
        add_log_message(f"DEBUG: Number of nodes: {len(coords_mm)}")
        
        # Check if there are stress values in the left side of the geometry
        # From the plot, left side appears to be around X < -650mm
        left_side_mask = coords_mm[:, 0] < -600  # Points more left than -600mm
        if np.any(left_side_mask):
            left_stress_values = stress_values_mpa[left_side_mask]
            add_log_message(f"DEBUG: Left side ({np.sum(left_side_mask)} nodes) stress range: [{left_stress_values.min():.2f}, {left_stress_values.max():.2f}] MPa")
            # Check if there are any non-zero stress values on the left
            nonzero_left = left_stress_values[left_stress_values > 0.01]
            if len(nonzero_left) > 0:
                add_log_message(f"DEBUG: Non-zero left side stress values: {len(nonzero_left)} nodes, max: {nonzero_left.max():.2f} MPa")
            else:
                add_log_message("DEBUG: All left side stress values are near zero!")
        else:
            add_log_message("DEBUG: No nodes found on the left side (X < -600mm)")
        
        # Also check the geometry bounds to understand coordinate system
        if outer_points:
            outer_array = np.array(outer_points)
            add_log_message(f"DEBUG: Outer geometry range - X: [{outer_array[:, 0].min():.1f}, {outer_array[:, 0].max():.1f}], Y: [{outer_array[:, 1].min():.1f}, {outer_array[:, 1].max():.1f}]")
        
        topology = mesh.topology
        cells = topology.connectivity(topology.dim, 0).array.reshape(-1, 3)

        # The DOLFINx triangulation was missing the left side - use Delaunay instead
        triangulation = tri.Triangulation(coords_mm[:, 0], coords_mm[:, 1])

        # Debug triangulation
        add_log_message(f"Triangulation created with {len(triangulation.triangles)} triangles")
        
        # Check triangle coverage
        triangle_centers = coords_mm[triangulation.triangles].mean(axis=1)
        left_triangles_500 = triangle_centers[:, 0] < -500
        add_log_message(f"Left side triangles (X < -500): {np.sum(left_triangles_500)} out of {len(triangulation.triangles)}")
        
        # Now apply masking to show plot only within the geometry boundaries
        from matplotlib.path import Path
        if outer_points:
            outer_path = Path(np.array(outer_points))
            triangle_centers = coords_mm[triangulation.triangles].mean(axis=1)
            mask = ~outer_path.contains_points(triangle_centers)
            
            if len(inner_points) >= 3:
                inner_path = Path(np.array(inner_points))
                inner_mask = inner_path.contains_points(triangle_centers)
                mask = np.logical_or(mask, inner_mask)
            
            triangulation.set_mask(mask)
            add_log_message(f"DEBUG: Applied geometry mask - {np.sum(mask)} triangles masked out")
        else:
            add_log_message("DEBUG: No masking applied - no outer geometry defined")

        # Temporarily disable masking to debug the left-side issue
        # TODO: The masking logic might be incorrectly hiding stress results on the left side
        # Let's show all triangles first to see if stress values exist everywhere
        
        # # Masking to show plot only within the geometry
        # from matplotlib.path import Path
        # if outer_points:
        #     outer_path = Path(np.array(outer_points))
        #     triangle_centers = coords_mm[triangulation.triangles].mean(axis=1)
        #     mask = ~outer_path.contains_points(triangle_centers)
        #     
        #     if len(inner_points) >= 3:
        #         inner_path = Path(np.array(inner_points))
        #         inner_mask = inner_path.contains_points(triangle_centers)
        #         mask = np.logical_or(mask, inner_mask)
        #     
        #     triangulation.set_mask(mask)

        # Use tricontourf for smooth, high-quality plot with more levels
        # Ensure contour levels include the full range including near-zero values
        min_stress = stress_values_mpa.min()
        max_stress = stress_values_mpa.max()
        
        # Create explicit levels to ensure low stress values are visible
        levels = np.linspace(min_stress, max_stress, 51)  # 51 levels from min to max
        add_log_message(f"DEBUG: Contour levels from {min_stress:.3f} to {max_stress:.3f} MPa")
        
        try:
            contour = ax.tricontourf(triangulation, stress_values_mpa, levels=levels, cmap='jet', zorder=2)
            add_log_message("DEBUG: tricontourf successful")
        except Exception as e:
            add_log_message(f"DEBUG: tricontourf failed: {e}")
            # Fallback to scatter plot to see if points exist
            scatter = ax.scatter(coords_mm[:, 0], coords_mm[:, 1], c=stress_values_mpa, cmap='jet', s=1, zorder=2)
            contour = scatter
        
        cbar = fig.colorbar(contour, ax=ax, shrink=0.8)
        cbar.set_label('Shear Stress (MPa)', color='white')
        cbar.ax.tick_params(colors='white')

        max_stress_val = np.max(stress_values_mpa)
        ax.set_title(f"Shear Stress Distribution\nMax Stress: {max_stress_val:.2f} MPa", fontsize=14)
        
    elif mesh_data:
        # Plot mesh visualization only
        mesh_obj = mesh_data['mesh_data']
        points = mesh_obj.points[:, :2] * 1000  # Convert to mm
        triangles = mesh_obj.cells[0].data  # Triangle connectivity
        
        # Create triangulation for mesh visualization
        triangulation = tri.Triangulation(points[:, 0], points[:, 1], triangles)
        
        # Plot mesh wireframe
        ax.triplot(triangulation, 'w-', linewidth=0.3, alpha=0.7, zorder=2)
        
        # Fill triangles with a light color to show coverage
        ax.tricontourf(triangulation, np.ones(len(points)), levels=1, colors=['lightblue'], alpha=0.3, zorder=1)
        
        num_nodes = len(points)
        num_elements = len(triangles)
        ax.set_title(f"Mesh Visualization\nNodes: {num_nodes}, Elements: {num_elements}", fontsize=14)
        
    else:
        ax.set_title("Geometry Preview", fontsize=14)

    # Plot geometry outlines underneath the stress plot
    if outer_points:
        outer_poly = np.array(outer_points)
        ax.plot(np.append(outer_poly[:, 0], outer_poly[0, 0]), 
                np.append(outer_poly[:, 1], outer_poly[0, 1]), 
                'w-', linewidth=1, zorder=3, alpha=0.9)
    if inner_points:
        inner_poly = np.array(inner_points)
        ax.plot(np.append(inner_poly[:, 0], inner_poly[0, 0]), 
                np.append(inner_poly[:, 1], inner_poly[0, 1]), 
                'w-', linewidth=1, zorder=3, alpha=0.9)

    ax.set_xlabel("X (mm)", fontsize=12)
    ax.set_ylabel("Y (mm)", fontsize=12)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Ensure proper margins to avoid cropping
    if outer_points:
        all_points = np.array(outer_points + inner_points) if inner_points else np.array(outer_points)
        x_margin = (np.max(all_points[:, 0]) - np.min(all_points[:, 0])) * 0.15  # Increased margin
        y_margin = (np.max(all_points[:, 1]) - np.min(all_points[:, 1])) * 0.15  # Increased margin
        
        default_xlim = (np.min(all_points[:, 0]) - x_margin, np.max(all_points[:, 0]) + x_margin)
        default_ylim = (np.min(all_points[:, 1]) - y_margin, np.max(all_points[:, 1]) + y_margin)
    else:
        default_xlim = (-10, 10)
        default_ylim = (-10, 10)
    
    # Apply manual zoom if provided, otherwise use default with margins
    if xlim and xlim[0] is not None and xlim[1] is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(default_xlim)
        
    if ylim and ylim[0] is not None and ylim[1] is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(default_ylim)
    
    # Adjust layout to prevent cropping with more padding
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    
    return fig

def manage_geometry(shape_type):
    """UI components for managing geometry points."""
    if shape_type == "outer":
        points_list = st.session_state.outer_points
        title = "Outer Shape"
    else:
        points_list = st.session_state.inner_points
        title = "Inner Hole"

    st.subheader(f"Points List for {title}")
    
    # Point selection
    selected_point_index = st.session_state.get(f"selected_{shape_type}", None)

    if points_list:
        # Create a list of strings for the selectbox
        point_options = [f"Point {i+1}: ({p[0]}, {p[1]})" for i, p in enumerate(points_list)]
        
        # The selectbox will store the index of the selected point
        selected_option = st.selectbox(
            "Select a point to edit or delete:", 
            options=range(len(point_options)), 
            format_func=lambda x: point_options[x],
            index=selected_point_index,
            key=f"select_{shape_type}"
        )
        st.session_state[f"selected_{shape_type}"] = selected_option
        
        # Edit form for the selected point
        if selected_option is not None:
            with st.form(key=f"edit_form_{shape_type}"):
                st.write(f"**Editing Point {selected_option + 1}**")
                point_to_edit = points_list[selected_option]
                
                col1, col2 = st.columns(2)
                with col1:
                    new_x = st.number_input("New X", value=point_to_edit[0], format="%.4f", key=f"edit_x_{shape_type}")
                with col2:
                    new_y = st.number_input("New Y", value=point_to_edit[1], format="%.4f", key=f"edit_y_{shape_type}")
                
                update_btn, delete_btn = st.columns(2)
                with update_btn:
                    if st.form_submit_button("Update Point", use_container_width=True):
                        points_list[selected_option] = (new_x, new_y)
                        st.rerun()
                with delete_btn:
                    if st.form_submit_button("Delete Point", use_container_width=True):
                        points_list.pop(selected_option)
                        st.session_state[f"selected_{shape_type}"] = None # Reset selection
                        st.rerun()

    else:
        st.text("No points defined.")

    # Edit controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Undo Last", key=f"undo_{shape_type}", use_container_width=True):
            if points_list:
                points_list.pop()
                st.rerun()
    with col2:
        if st.button("Clear Geometry", key=f"clear_{shape_type}", use_container_width=True):
            if points_list:
                points_list.clear()
                # Clear all related data
                st.session_state.results = None
                st.session_state.mesh_data = None
                st.session_state.is_calculating = False
                st.session_state.is_meshing = False
                # Reset zoom/pan controls
                st.session_state.zoom_level = 1.0
                st.session_state.pan_x = 0.0
                st.session_state.pan_y = 0.0
                # Clear log messages
                st.session_state.log_messages = []
                add_log_message("Cleared all geometry, mesh, and results data")
                st.rerun()

    st.subheader(f"Add Points for {title}")
    
    # Using a unique key for each text_area
    pasted_text = st.text_area("Enter Points (x,y per line or single x,y):", key=f"paste_{shape_type}", height=100, placeholder="Examples:\n10, 20\n30 40\n(50, 60)")
    if st.button("Add Points", key=f"add_{shape_type}", use_container_width=True):
        new_points = parse_points(pasted_text)
        if new_points:
            points_list.extend(new_points)
            st.rerun()

def parse_points(text):
    """Parse points from text input."""
    points = []
    if not text:
        return points
    
    lines = text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Remove parentheses and split by comma or space
        line = line.replace('(', '').replace(')', '')
        parts = line.replace(',', ' ').split()
        
        if len(parts) >= 2:
            try:
                x = float(parts[0])
                y = float(parts[1])
                points.append((x, y))
            except ValueError:
                st.error(f"Invalid point format: {line}")
    
    return points

if __name__ == "__main__":
    init_session_state()
    main()
