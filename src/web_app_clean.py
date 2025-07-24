import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import logging
import os
from fea import meshing, solver
from fea.multi_method_analysis import MultiMethodAnalysis
from fea.mixed_section_analysis import MixedSectionAnalysis

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
    # Add state for plot caching
    if 'cached_plot_data' not in st.session_state: st.session_state.cached_plot_data = None
    if 'plot_cache_key' not in st.session_state: st.session_state.plot_cache_key = None

def add_log_message(message):
    """Add a message to the log with timestamp"""
    import datetime
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    st.session_state.log_messages.append(full_message)
    # Keep only last 50 messages to prevent memory issues
    if len(st.session_state.log_messages) > 50:
        st.session_state.log_messages = st.session_state.log_messages[-50:]

def clear_previous_results():
    """Clear previous mesh data and results to prevent loops and stale data"""
    add_log_message("🧹 Clearing previous mesh and results...")
    
    # Clear mesh data
    if 'mesh_data' in st.session_state:
        st.session_state.mesh_data = None
    
    # Clear results
    if 'results' in st.session_state:
        st.session_state.results = None
    
    # Reset calculation states
    st.session_state.is_calculating = False
    st.session_state.is_meshing = False
    
    # Clear any cached plot data
    if 'plot_data' in st.session_state:
        st.session_state.plot_data = None
    
    # Clear plot cache to force regeneration
    if 'cached_plot_data' in st.session_state:
        st.session_state.cached_plot_data = None
        st.session_state.plot_cache_key = None
    
    add_log_message("✅ Previous data cleared - ready for new calculation")

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
    
    # Analysis method toggle at the top
    st.subheader("🔧 Analysis Method")
    analysis_method = st.radio(
        "Choose analysis method:",
        ["Mixed Method (Thin Structures)", "Saint-Venant FEA"],
        index=0,
        help="Mixed Method is recommended for thin-walled sections, FEA for thick/solid sections"
    )

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
        
        # Update session state - NO st.rerun() to avoid app refresh
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
        
        # Wall thickness (for mixed method analysis)
        wall_thickness_mm = st.slider("Wall Thickness (mm)", 
                                     min_value=1.0, max_value=20.0, 
                                     value=3.0, step=0.5,
                                     help="Actual wall thickness of the structure (used for Mixed Method)")
        
        # === SANDWICH CONSTRUCTION SECTION ===
        st.subheader("🥪 Construction Type")
        construction_type = st.radio("Select construction:", 
                                   ["Thin-wall (current geometry)", "Sandwich construction"],
                                   help="Thin-wall uses current coordinates as-is. Sandwich calculates effective properties for laminated construction.")
        
        if construction_type == "Sandwich construction":
            st.info("📐 **Note**: Your current geometry coordinates will be treated as the mid-plane. The app will calculate effective properties for the specified sandwich thickness.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Skin Properties:**")
                skin_thickness = st.number_input("Skin thickness (each side) [mm]:", 
                                               value=1.5, min_value=0.5, max_value=10.0, step=0.1,
                                               help="Thickness of each composite skin")
                skin_material = st.selectbox("Skin material:", 
                                           ["Carbon Fiber Quasi-iso", "Carbon Fiber Conservative", "Glass Fiber"],
                                           help="Skin material determines shear modulus")
            
            with col2:
                st.write("**Core Properties:**")
                core_thickness = st.number_input("Core thickness [mm]:", 
                                               value=15.0, min_value=5.0, max_value=50.0, step=1.0,
                                               help="Thickness of foam/honeycomb core")
                core_material = st.selectbox("Core material:", 
                                           ["PVC Foam (80 kg/m³)", "PVC Foam (100 kg/m³)", "Honeycomb"],
                                           help="Core material affects shear transfer")
            
            # Calculate effective properties
            def calculate_sandwich_properties(skin_thickness, core_thickness, skin_material, core_material):
                # Skin material properties (GPa)
                skin_G_map = {
                    "Carbon Fiber Quasi-iso": 20.0,
                    "Carbon Fiber Conservative": 5.5, 
                    "Glass Fiber": 3.5
                }
                
                # Core material properties (MPa)
                core_G_map = {
                    "PVC Foam (80 kg/m³)": 50.0,
                    "PVC Foam (100 kg/m³)": 80.0,
                    "Honeycomb": 200.0
                }
                
                skin_thickness_m = skin_thickness * 1e-3
                core_thickness_m = core_thickness * 1e-3
                total_thickness = 2 * skin_thickness_m + core_thickness_m
                
                # Volume fractions
                skin_fraction = (2 * skin_thickness_m) / total_thickness
                core_fraction = core_thickness_m / total_thickness
                
                # Material properties
                G_skin = skin_G_map[skin_material] * 1e9  # Convert to Pa
                G_core = core_G_map[core_material] * 1e6   # Convert to Pa
                
                # Effective shear modulus for torsion (skins carry most load)
                shear_efficiency = 0.1  # Core contributes ~10% for torsion
                G_effective = skin_fraction * G_skin + core_fraction * G_core * shear_efficiency
                
                return {
                    'total_thickness_mm': total_thickness * 1000,
                    'skin_fraction': skin_fraction,
                    'core_fraction': core_fraction,
                    'G_effective_GPa': G_effective / 1e9,
                    'G_effective_Pa': G_effective
                }
            
            # Calculate and display properties
            sandwich_props = calculate_sandwich_properties(skin_thickness, core_thickness, skin_material, core_material)
            
            # Display results
            st.write("**📊 Calculated Properties:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Thickness", f"{sandwich_props['total_thickness_mm']:.1f} mm")
                st.metric("Skin Volume %", f"{sandwich_props['skin_fraction']*100:.1f}%")
            with col2:
                st.metric("Effective G", f"{sandwich_props['G_effective_GPa']:.2f} GPa")
                # Calculate improvement over thin-wall material
                base_G = 5.5 if skin_material == "Carbon Fiber Conservative" else 20.0 if skin_material == "Carbon Fiber Quasi-iso" else 3.5
                G_improvement = sandwich_props['G_effective_GPa'] / base_G
                st.metric("vs. Skin G", f"{G_improvement:.2f}x")
            with col3:
                st.metric("Core Volume %", f"{sandwich_props['core_fraction']*100:.1f}%")
                if sandwich_props['G_effective_GPa'] > 10:
                    st.success("🎯 High stiffness!")
                elif sandwich_props['G_effective_GPa'] > 5:
                    st.info("✅ Good properties")
                else:
                    st.warning("⚠️ Consider upgrade")
            
            # Use calculated effective modulus
            g_modulus = sandwich_props['G_effective_Pa'] / 1e6  # Convert to MPa for input
            
        else:
            # Traditional thin-wall construction
            g_modulus = st.number_input("Shear Modulus (G) [MPa]:", value=20e3, format="%e", 
                                       help="CF quasi-iso: ~20 GPa, CF conservative: ~5.5 GPa, Steel: ~80 GPa")
            
            # Material presets (only for thin-wall)
            st.write("**Quick Material Presets:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("CF Quasi-iso", help="Carbon fiber in-plane shear (20 GPa)"):
                    g_modulus = 20e3  # 20 GPa
                    st.rerun()
            with col2:
                if st.button("CF Conservative", help="Carbon fiber through-thickness (5.5 GPa)"):
                    g_modulus = 5.5e3  # 5.5 GPa
                    st.rerun()
            with col3:
                if st.button("Steel", help="Structural steel (80 GPa)"):
                    g_modulus = 80e3  # 80 GPa  
                    st.rerun()
        torque = st.number_input("Applied Torque (T) [N-m]:", value=1000.0)
        length = st.number_input("Beam Length (L) [m]:", value=2.0)

        # Check if any operation is currently in progress
        is_busy = getattr(st.session_state, 'is_calculating', False) or getattr(st.session_state, 'is_meshing', False)
        
        if st.button("Generate Mesh & Solve", type="primary", use_container_width=True, disabled=is_busy):
            # Clear previous data to prevent loops and stale results
            clear_previous_results()
            st.session_state.is_calculating = True
            add_log_message("Starting complete analysis...")
            # Store refinement setting for the analysis
            st.session_state.refinement_levels = refinement_levels
            # Store sandwich construction parameters
            st.session_state.construction_type = construction_type
            if construction_type == "Sandwich construction":
                st.session_state.sandwich_props = sandwich_props
            st.rerun()

        # Separate mesh generation button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("1️⃣ Generate Mesh", use_container_width=True, disabled=is_busy):
                # Clear previous data to prevent mesh conflicts
                clear_previous_results()
                st.session_state.is_meshing = True
                add_log_message("Starting mesh generation...")
                st.session_state.refinement_levels = refinement_levels
                st.rerun()
        
        with col2:
            mesh_available = st.session_state.mesh_data is not None
            if st.button("2️⃣ Solve FEA", use_container_width=True, disabled=(not mesh_available or is_busy)):
                if mesh_available:
                    # Clear previous results but keep mesh data
                    if 'results' in st.session_state:
                        st.session_state.results = None
                    st.session_state.is_calculating = True   # Set for new calculation
                    add_log_message("Starting FEA solve with existing mesh...")
                    # Store sandwich construction parameters for solve-only
                    st.session_state.construction_type = construction_type
                    if construction_type == "Sandwich construction":
                        st.session_state.sandwich_props = sandwich_props
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
            try:
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
                    # Clear plot cache since we have new results
                    st.session_state.cached_plot_data = None
                    st.session_state.plot_cache_key = None
                    add_log_message("Analysis completed successfully!")
                    # Add debug info for total deflection
                    if results.get('u') is not None and results.get('theta') is not None:
                        u = results['u']
                        theta = results['theta']
                        coords = u.function_space.tabulate_dof_coordinates()[:, :2]
                        coords_mm = coords * 1000
                        
                        # Calculate maximum radial distance and total deflection
                        centroid_x = np.mean(coords_mm[:, 0])
                        centroid_y = np.mean(coords_mm[:, 1])
                        dx = coords_mm[:, 0] - centroid_x
                        dy = coords_mm[:, 1] - centroid_y
                        max_radius_mm = np.sqrt(dx**2 + dy**2).max()
                        max_total_deflection_mm = theta * max_radius_mm
                        
                        add_log_message(f"DEBUG: Max total deflection: {max_total_deflection_mm:.2f} mm")
                        add_log_message(f"DEBUG: Max radius from centroid: {max_radius_mm:.2f} mm")
                else:
                    add_log_message("Analysis failed. Please check geometry and parameters.")
            except Exception as e:
                add_log_message(f"Error during calculation: {str(e)}")
                st.session_state.results = None
            finally:
                # Always reset the calculating flag to prevent loops
                st.session_state.is_calculating = False
        
        # Conditional plot based on analysis method
        if analysis_method == "Mixed Method (Thin Structures)" and st.session_state.results:
            # Show mixed method stress plot
            try:
                outer_points_m = [(p[0]/1000, p[1]/1000) for p in st.session_state.outer_points]
                inner_points_m = [(p[0]/1000, p[1]/1000) for p in st.session_state.inner_points] if st.session_state.inner_points else []
                wall_thickness_m = wall_thickness_mm / 1000
                
                mixed_analyzer = MixedSectionAnalysis(outer_points_m, inner_points_m, wall_thickness_m)
                mixed_results = mixed_analyzer.analyze_mixed_section(
                    G=g_modulus * 1e6,  # Convert MPa to Pa for proper units
                    T=torque,
                    L=length
                )
                
                if 'total' in mixed_results and mixed_results['total']:
                    stress_fig = mixed_analyzer.create_stress_plot(mixed_results, torque)
                    st.pyplot(stress_fig, use_container_width=True)
                else:
                    st.error("Mixed section analysis failed - showing FEA plot")
                    fig = plot_stress_distribution(
                        st.session_state.results,
                        st.session_state.outer_points,
                        st.session_state.inner_points,
                        xlim=(st.session_state.x_min, st.session_state.x_max),
                        ylim=(st.session_state.y_min, st.session_state.y_max),
                        mesh_data=st.session_state.mesh_data
                    )
                    st.pyplot(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Mixed method plot failed: {e} - showing FEA plot")
                fig = plot_stress_distribution(
                    st.session_state.results,
                    st.session_state.outer_points,
                    st.session_state.inner_points,
                    xlim=(st.session_state.x_min, st.session_state.x_max),
                    ylim=(st.session_state.y_min, st.session_state.y_max),
                    mesh_data=st.session_state.mesh_data
                )
                st.pyplot(fig, use_container_width=True)
        else:
            # Show FEA stress plot
            fig = plot_stress_distribution(
                st.session_state.results,
                st.session_state.outer_points,
                st.session_state.inner_points,
                xlim=(st.session_state.x_min, st.session_state.x_max),
                ylim=(st.session_state.y_min, st.session_state.y_max),
                mesh_data=st.session_state.mesh_data
            )
            st.pyplot(fig, use_container_width=True)

    with col2:
        st.header("Results")
        if st.session_state.results:
            results = st.session_state.results
            
            # Conditional Analysis based on toggle
            if analysis_method == "Mixed Method (Thin Structures)":
                # Mixed Section Analysis
                st.subheader("🎯 Mixed Section Analysis")
                try:
                    # Get current geometry
                    outer_points_m = [(p[0]/1000, p[1]/1000) for p in st.session_state.outer_points]
                    inner_points_m = [(p[0]/1000, p[1]/1000) for p in st.session_state.inner_points] if st.session_state.inner_points else []
                    
                    wall_thickness_m = wall_thickness_mm / 1000
                    
                    # Run mixed section analysis
                    mixed_analyzer = MixedSectionAnalysis(outer_points_m, inner_points_m, wall_thickness_m)
                    mixed_results = mixed_analyzer.analyze_mixed_section(
                        G=g_modulus * 1e6,  # Convert MPa to Pa for proper units
                        T=torque,
                        L=length
                    )
                    
                    # Display mixed section results in standardized format
                    if 'total' in mixed_results and mixed_results['total']:
                        total = mixed_results['total']
                        geometry = mixed_results.get('geometry', {})
                        
                        # Get area from geometry (should be in m²)
                        area_m2 = geometry.get('area', 0)
                        if area_m2 == 0:
                            # Fallback: calculate area from FEA results
                            area_m2 = results.get('area', 0)
                        area_mm2 = area_m2 * 1e6  # Convert m² to mm²
                        
                        # Standardized metrics format (same as FEA)
                        st.metric("Cross-sectional Area", f"{area_mm2:.2f} mm²")
                        
                        J_mm4 = total['J'] * 1e12  # Convert m⁴ to mm⁴
                        st.metric("Torsional Constant (J)", f"{J_mm4:.4e} mm⁴")
                        
                        # UNIT CONVERSION FIX: Convert G from MPa to Pa for proper units
                        g_modulus_pa = g_modulus * 1e6  # Convert MPa to Pa (N/m²)
                        k_value = total.get('k', total['J'] * g_modulus_pa)  # G*J in N⋅m/rad (J in m⁴, G in N/m²)
                        st.metric("Torsional Stiffness (k)", f"{k_value:.4e} N⋅m/rad")
                        
                        # Calculate angle: θ = TL/(GJ) in radians, then convert to degrees
                        theta_rad_corrected = torque * length / k_value
                        theta_deg_corrected = np.rad2deg(theta_rad_corrected)
                        st.metric("Angle of Twist (θ)", f"{theta_deg_corrected:.6f} deg")
                        
                        # Get stress data from mixed_results
                        stress_data = mixed_results.get('stress_analysis', {})
                        max_stress_mpa = stress_data.get('max_stress_value', 0) / 1e6 if stress_data.get('max_stress_value') else 0
                        st.metric("Max Shear Stress", f"{max_stress_mpa:.2f} MPa")
                        
                        # FIXED: Calculate deflection using radians (not degrees)
                        # Maximum deflection = θ (radians) × max_radius
                        if st.session_state.outer_points:
                            coords_mm = np.array(st.session_state.outer_points)
                            centroid_x = np.mean(coords_mm[:, 0])
                            centroid_y = np.mean(coords_mm[:, 1])
                            dx = coords_mm[:, 0] - centroid_x
                            dy = coords_mm[:, 1] - centroid_y
                            max_radius_mm = np.sqrt(dx**2 + dy**2).max()
                            max_deflection_mm = theta_rad_corrected * max_radius_mm  # Use radians!
                            st.metric("Max Total Deflection", f"{max_deflection_mm:.2f} mm")
                        
                        # Engineering guidance
                        if max_stress_mpa > 0:
                            st.info(f"""
                            **🔥 Peak Stress Analysis:**
                            - Maximum shear stress: **{max_stress_mpa:.2f} MPa**
                            - Wall thickness: {wall_thickness_mm} mm
                            
                            **Engineering Notes:**
                            - Closed sections have constant stress around perimeter
                            - Open sections have peak stress at constraint points
                            - Consider local reinforcement at high-stress areas
                            """)
                        
                        # Engineering recommendation
                        thickness_ratio = wall_thickness_m / 1.4  # Approximate characteristic dimension
                        if thickness_ratio < 0.01:
                            st.success("✅ **Analysis Validity**: Excellent for this very thin-walled geometry")
                        elif thickness_ratio < 0.05:
                            st.success("✅ **Analysis Validity**: Good for this thin-walled geometry")
                        else:
                            st.warning("⚠️ **Analysis Validity**: Consider FEA method for thicker sections")
                    
                    else:
                        st.error("Mixed section analysis failed")
                        
                except Exception as e:
                    st.error(f"Mixed section analysis failed: {e}")
            
            else:  # Saint-Venant FEA
                # FEA Method Analysis
                st.subheader("🔬 Saint-Venant FEA Analysis")
                
                # Display FEA results in standardized format
                area_mm2 = results['area'] * 1e6  # Convert m² to mm²
                st.metric("Cross-sectional Area", f"{area_mm2:.2f} mm²")
                
                J_mm4 = results['J'] * 1e12  # Convert m⁴ to mm⁴ 
                st.metric("Torsional Constant (J)", f"{J_mm4:.4e} mm⁴")
                
                st.metric("Torsional Stiffness (k)", f"{results['k']:.4e} N⋅m/rad")
                st.metric("Angle of Twist (θ)", f"{np.rad2deg(results['theta']):.6f} deg")
                st.metric("Max Shear Stress", f"{results['tau_max']/1e6:.2f} MPa")
                
                # FIXED: Calculate deflection using radians (not degrees)
                # Maximum deflection = θ (radians) × max_radius
                if st.session_state.outer_points:
                    coords_mm = np.array(st.session_state.outer_points)
                    centroid_x = np.mean(coords_mm[:, 0])
                    centroid_y = np.mean(coords_mm[:, 1])
                    dx = coords_mm[:, 0] - centroid_x
                    dy = coords_mm[:, 1] - centroid_y
                    max_radius_mm = np.sqrt(dx**2 + dy**2).max()
                    max_deflection_mm = results['theta'] * max_radius_mm  # Use radians!
                    st.metric("Max Total Deflection", f"{max_deflection_mm:.2f} mm")
                
                # === ENGINEERING ASSESSMENT ===
                st.subheader("🔧 Engineering Assessment")
                
                # Get construction type for proper assessment
                construction_type = getattr(st.session_state, 'construction_type', "Thin-wall (current geometry)")
                
                # Twist angle assessment
                twist_deg = np.rad2deg(results['theta'])
                if twist_deg < 1.0:
                    twist_status = "🎯 **EXCELLENT** - Suitable for precision applications"
                elif twist_deg < 5.0:
                    twist_status = "✅ **GOOD** - Acceptable for most structural applications"
                elif twist_deg < 15.0:
                    twist_status = "⚠️ **MARGINAL** - May need stiffening for critical loads"
                else:
                    twist_status = "❌ **POOR** - Requires structural redesign or stiffening"
                
                # J value assessment based on construction type
                if construction_type == "Sandwich construction":
                    j_status = "✅ **SANDWICH** - Enhanced properties applied"
                else:
                    # Thin-wall assessment
                    thickness_ratio = wall_thickness_mm / 1400  # Approximate thickness ratio
                    if thickness_ratio < 0.01:
                        j_status = "⚠️ **THIN-WALL** - FEA may overestimate stiffness for very thin sections"
                    elif thickness_ratio < 0.05:
                        j_status = "ℹ️ **THIN-WALL** - Consider Mixed Method for comparison"
                    else:
                        j_status = "✅ **APPROPRIATE** - FEA suitable for this thickness ratio"
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Twist Performance:**")
                    st.write(twist_status)
                    st.write(f"Angle: {twist_deg:.3f}° for {torque:.0f} Nm")
                
                with col2:
                    st.write("**Structural Analysis:**")
                    st.write(j_status)
                    if construction_type == "Sandwich construction":
                        st.write(f"Construction: {construction_type}")
                    else:
                        st.write(f"Wall thickness: {wall_thickness_mm:.1f} mm")
            
            # Full Multi-Method Analysis (in expander)
            with st.expander("🔬 Detailed Multi-Method Comparison", expanded=False):
                try:
                    # Get current geometry
                    outer_points_m = [(p[0]/1000, p[1]/1000) for p in st.session_state.outer_points]
                    inner_points_m = [(p[0]/1000, p[1]/1000) for p in st.session_state.inner_points] if st.session_state.inner_points else []
                    
                    # Initialize multi-method analysis
                    multi_analysis = MultiMethodAnalysis(outer_points_m, inner_points_m)
                    
                    # Run all methods with current parameters
                    multi_results = multi_analysis.analyze_all_methods(
                        G=g_modulus,  # Already in Pa from input conversion
                        T=torque,
                        L=length,
                        fea_J=results['J']
                    )
                    
                    # Display comparison table
                    comparison_data = []
                    for method_name, method_data in multi_results.items():
                        if method_name in ['geometry', 'comparison']:
                            continue
                        
                        if 'J' in method_data and method_data['J'] is not None:
                            comparison_data.append({
                                'Method': method_data.get('method', method_name.upper()),
                                'J (m⁴)': f"{method_data['J']:.4e}",
                                'J (mm⁴)': f"{method_data['J']*1e12:.4e}",
                                'θ (deg)': f"{method_data.get('theta_deg', 0):.3f}",
                                'τ_max (MPa)': f"{method_data.get('max_shear_stress', 0)/1e6:.1f}"
                            })
                        else:
                            comparison_data.append({
                                'Method': method_data.get('method', method_name.upper()),
                                'J (m⁴)': method_data.get('note', 'Failed'),
                                'J (mm⁴)': '-',
                                'θ (deg)': '-',
                                'τ_max (MPa)': '-'
                            })
                    
                    if comparison_data:
                        df = pd.DataFrame(comparison_data)
                        st.dataframe(df, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Multi-method analysis failed: {e}")
                    st.info("Showing basic FEA results only.")
            
            # Add debugging information
            st.info(f"""
            **🔍 Calculation Check:**
            - Area = {results['area']:.6f} m² = {results['area']*1e6:.2f} mm²
            - J = {results['J']:.4e} m⁴
            - G = {g_modulus:.0f} MPa = {g_modulus*1e6:.4e} Pa  
            - L = {length:.1f} m
            - T = {torque:.0f} Nm
            - k = G×J/L = {results['k']:.4e} Nm/rad
            - θ = T/k = {np.rad2deg(results['theta']):.2f}° = {results['theta']:.4f} rad
            """)
            
            # Add solver information with explanation
            st.info("""
            **ℹ️ Understanding the Results:**
            
            • **Angle of Twist (θ)**: Total angular rotation of the shaft over its length L
            • **Total Deflection**: Circumferential displacement = θ × radius from center
            • **Shear Stress**: In-plane stress causing the torsion
            
            **Beam Constraints:**
            • One end fixed, other end has applied torque T = {T} Nm
            • Cross-section free to warp (no constraint on out-of-plane displacement)
            • Analysis assumes uniform twist along length L = {L} m
            
            **Note**: This is a 2D cross-sectional analysis using classical torsion theory.
            
            This web app uses **DOLFINx FEA** which solves the full torsion equations with proper boundary conditions.
            
            *Note: Results may differ from simplified analytical methods that assume circular shaft behavior.*
            """)
            
            # Add expandable physics explanation
            with st.expander("📚 Understanding Torsion Results"):
                st.markdown("""
                **Warping Visualization Explained:**
                
                The warping visualization shows how the cross-section deforms out-of-plane during torsion:
                
                **What you're seeing:**
                - **Contour Colors**: Show the magnitude of out-of-plane displacement
                - **Red Circles**: Points that move outward from the cross-section plane (+)
                - **Blue Circles**: Points that move inward from the cross-section plane (-)
                - **Circle Size**: Proportional to the amount of warping displacement
                
                **Physical Meaning:**
                - Non-circular cross-sections cannot remain flat during torsion
                - Different parts of the cross-section move different amounts perpendicular to the plane
                - This warping allows the development of shear stresses needed to carry torque
                - The warping pattern is unique to each cross-section geometry
                
                **Key Insight:**
                - Circular sections don't warp (they remain flat)
                - Rectangular and complex sections warp significantly
                - Warping enables non-circular sections to carry torsional loads
                - The warping displacements are typically very small but structurally important
                
                This visualization helps understand why torsional behavior differs between circular and non-circular cross-sections!
                
                **Note**: For a full 3D solid analysis, you'd need a 3D mesh with proper end constraints, 
                but that would be computationally intensive for a web app.
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
        
        J, k, theta, tau_max, tau_magnitude, V_mag, u, area = solver.solve_torsion(mesh_dir, G_Pa, T, L)
        
        add_log_message("Calculating torsional properties...")
        add_log_message(f"Cross-sectional area = {area:.6f} m² = {area*1e6:.2f} mm²")
        add_log_message(f"Torsional constant J = {J:.4e} m⁴")
        add_log_message(f"Torsional stiffness k = {k:.4e} Nm/rad")
        add_log_message(f"Angle of twist θ = {np.rad2deg(theta):.4f} degrees")
        add_log_message(f"Maximum shear stress = {tau_max/1e6:.2f} MPa")
        add_log_message("Post-processing stress field...")
        
        return {
            "J": J, "k": k, "theta": theta, "tau_max": tau_max,
            "tau_magnitude": tau_magnitude, "V_mag": V_mag,
            "u": u, "area": area
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
        
        J, k, theta, tau_max, tau_magnitude, V_mag, u, area = solver.solve_torsion(mesh_dir, G_Pa, T, L)
        
        add_log_message("Calculating torsional properties...")
        add_log_message(f"Cross-sectional area = {area:.6f} m² = {area*1e6:.2f} mm²")
        add_log_message(f"Torsional constant J = {J:.4e} m⁴")
        add_log_message(f"Torsional stiffness k = {k:.4e} Nm/rad")
        add_log_message(f"Angle of twist θ = {np.rad2deg(theta):.4f} degrees")
        add_log_message(f"Maximum shear stress = {tau_max/1e6:.2f} MPa")
        add_log_message("Post-processing stress field...")
        
        return {
            "J": J, "k": k, "theta": theta, "tau_max": tau_max,
            "tau_magnitude": tau_magnitude, "V_mag": V_mag,
            "u": u, "area": area
        }
        
    except Exception as e:
        add_log_message(f"ERROR: FEA solve failed - {str(e)}")
        return None

def plot_stress_distribution(results, outer_points, inner_points, xlim=None, ylim=None, mesh_data=None):
    """
    Plots the stress distribution using Matplotlib for high-quality rendering.
    If only mesh_data is provided (no results), shows mesh visualization.
    Uses caching to improve zoom/pan performance.
    """
    
    # Create cache key for expensive operations
    cache_key = None
    if results and results.get("tau_magnitude") is not None:
        # Simple cache key based on results object id and geometry
        cache_key = f"{id(results.get('tau_magnitude'))}_{len(outer_points)}_{len(inner_points)}"
        
        # Check if we can reuse cached triangulation data
        if (hasattr(st.session_state, 'plot_cache_key') and 
            st.session_state.plot_cache_key == cache_key and
            st.session_state.cached_plot_data is not None):
            
            # Use cached data
            cached_data = st.session_state.cached_plot_data
            coords_mm = cached_data['coords_mm']
            stress_values_mpa = cached_data['stress_values_mpa']
            triangulation = cached_data['triangulation']
            max_stress_val = cached_data['max_stress_val']
            use_cached = True
        else:
            use_cached = False
    else:
        use_cached = False
    
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

    # Plot stress results
    if results and results.get("tau_magnitude") is not None and results.get("V_mag") is not None:
        
        if not use_cached:
            # Expensive operations - only do once and cache
            tau_magnitude = results["tau_magnitude"]
            V_mag = results["V_mag"]
            mesh = V_mag.mesh
            coords = V_mag.tabulate_dof_coordinates()[:, :2]
            stress_values_pa = tau_magnitude.x.array
            coords_mm = coords * 1000
            stress_values_mpa = stress_values_pa / 1e6
            
            # Reduced debug logging - only log summary info
            add_log_message(f"📊 Mesh: {len(coords_mm)} nodes, Stress: {stress_values_mpa.min():.2f}-{stress_values_mpa.max():.2f} MPa")
            
            # Create triangulation (expensive operation)
            triangulation = tri.Triangulation(coords_mm[:, 0], coords_mm[:, 1])
            max_stress_val = np.max(stress_values_mpa)
            
            # Cache the expensive data
            st.session_state.cached_plot_data = {
                'coords_mm': coords_mm,
                'stress_values_mpa': stress_values_mpa,
                'triangulation': triangulation,
                'max_stress_val': max_stress_val
            }
            st.session_state.plot_cache_key = cache_key
            add_log_message("💾 Plot data cached for fast zoom/pan")
        else:
            add_log_message("⚡ Using cached plot data for fast rendering")
        
        # Apply geometry masking (relatively fast operation)
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

        # Create contour plot with optimized levels
        min_stress = stress_values_mpa.min()
        max_stress = stress_values_mpa.max()
        levels = np.linspace(min_stress, max_stress, 31)  # Reduced from 51 to 31 levels
        
        try:
            contour = ax.tricontourf(triangulation, stress_values_mpa, levels=levels, cmap='jet', zorder=2)
        except Exception as e:
            add_log_message(f"⚠️ Contour plot failed: {e} - using scatter plot")
            scatter = ax.scatter(coords_mm[:, 0], coords_mm[:, 1], c=stress_values_mpa, cmap='jet', s=1, zorder=2)
            contour = scatter
        
        cbar = fig.colorbar(contour, ax=ax, shrink=0.8)
        cbar.set_label('Shear Stress (MPa)', color='white')
        cbar.ax.tick_params(colors='white')

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
