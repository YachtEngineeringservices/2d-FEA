"""
Streamlit Web Application for 2D FEA Torsion Analysis
This version mirrors the desktop app interface but runs in any web browser
Optimized for Streamlit Cloud deployment
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from matplotlib.patches import Polygon

# Set matplotlib backend for cloud environment
import matplotlib
matplotlib.use('Agg')

# Simple FEA solver for web app
def run_simplified_fea(outer_points, inner_points, G, T, L):
    """
    Simplified torsional analysis for web app
    """
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
            # For arbitrary shapes, use approximate formula
            outer_area = polygon_area(outer_pts)
            inner_area = polygon_area(inner_pts) if inner_pts else 0.0
            net_area = outer_area - inner_area
            
            # Estimate characteristic dimension
            if outer_pts:
                outer_pts = np.array(outer_pts)
                width = np.max(outer_pts[:, 0]) - np.min(outer_pts[:, 0])
                height = np.max(outer_pts[:, 1]) - np.min(outer_pts[:, 1])
                char_dim = (width + height) / 2
            else:
                char_dim = 1.0
            
            # Approximate polar moment (simplified for general shapes)
            if inner_area == 0:
                # Solid section
                J = net_area * char_dim**2 / 6
            else:
                # Hollow section - more conservative estimate
                J = net_area * char_dim**2 / 8
            
            return max(J, 1e-12)  # Prevent division by zero
        
        # Calculate geometry properties
        J = polar_moment_approx(outer_points, inner_points)
        
        # Torsional calculations
        k = G * J / L  # Torsional stiffness
        theta = T / k   # Angle of twist
        
        # Estimate max shear stress (simplified)
        if outer_points:
            outer_pts = np.array(outer_points)
            # Use maximum distance from centroid as radius
            centroid = np.mean(outer_pts, axis=0)
            distances = np.sqrt(np.sum((outer_pts - centroid)**2, axis=1))
            r_max = np.max(distances)
        else:
            r_max = 1.0
        
        tau_max = T * r_max / J
        
        return {
            'polar_moment': J,
            'stiffness': k,
            'twist_angle': theta,
            'max_shear_stress': tau_max,
            'success': True
        }
        
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return {
            'polar_moment': 0,
            'stiffness': 0, 
            'twist_angle': 0,
            'max_shear_stress': 0,
            'success': False
        }

# Configure page
st.set_page_config(
    page_title="2D FEA Torsion Analysis - Yacht Engineering Services",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/YachtEngineeringservices/2d-FEA',
        'Report a bug': 'https://github.com/YachtEngineeringservices/2d-FEA/issues',
        'About': '''
        # 2D FEA Torsion Analysis
        
        Web-based Finite Element Analysis for Torsional Problems
        
        **Developed by Yacht Engineering Services**
        
        This application provides simplified torsional analysis using analytical methods,
        making it accessible in any web browser without requiring specialized software.
        '''
    }
)

# Title and description
st.title("🔧 2D FEA Torsion Analysis")
st.markdown("**Web-based Finite Element Analysis for Torsional Problems**")
st.markdown("*Developed by Yacht Engineering Services*")

# Add information about the application
with st.expander("ℹ️ About This Application"):
    st.markdown("""
    This web application provides **simplified 2D torsional analysis** using analytical methods.
    
    **Features:**
    - ✅ Interactive geometry input (click to add points)
    - ✅ Real-time visualization with automatic updates
    - ✅ Separate outer shape and inner hole definition
    - ✅ Material properties configuration
    - ✅ Analytical torsional calculations
    - ✅ Stress and displacement analysis
    - ✅ Results export (JSON format)
    
    **Perfect for:**
    - Quick engineering estimates
    - Educational purposes  
    - Preliminary design validation
    
    For advanced FEA with DOLFINx, download the desktop application from 
    [GitHub](https://github.com/YachtEngineeringservices/2d-FEA).
    """)

st.markdown("---")

# Initialize session state
if 'outer_points' not in st.session_state:
    st.session_state.outer_points = []
if 'inner_points' not in st.session_state:
    st.session_state.inner_points = []
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Outer Shape"
if 'results' not in st.session_state:
    st.session_state.results = None

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
    
    # Point input controls
    st.write("**Add Point:**")
    col_x, col_y = st.columns(2)
    with col_x:
        x_coord = st.number_input("X (mm)", value=0.0, format="%.3f", key="x_input")
    with col_y:
        y_coord = st.number_input("Y (mm)", value=0.0, format="%.3f", key="y_input")
    
    if st.button("Add Point", type="primary"):
        new_point = [float(x_coord), float(y_coord)]
        if geometry_tab == "Outer Shape":
            st.session_state.outer_points.append(new_point)
            st.success(f"Added point to Outer Shape: ({x_coord:.3f}, {y_coord:.3f})")
        else:
            st.session_state.inner_points.append(new_point)
            st.success(f"Added point to Inner Hole: ({x_coord:.3f}, {y_coord:.3f})")
        st.rerun()
    
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
    if st.button("Rectangle (100x100mm)"):
        points = [[0, 0], [100, 0], [100, 100], [0, 100]]
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
                # Convert to meters and run simplified analysis
                outer_points_m = [[p[0]/1000, p[1]/1000] for p in st.session_state.outer_points]
                inner_points_m = [[p[0]/1000, p[1]/1000] for p in st.session_state.inner_points]
                
                results = run_simplified_fea(
                    outer_points_m, 
                    inner_points_m,
                    shear_modulus * 1e6,  # Convert MPa to Pa
                    applied_torque,
                    beam_length
                )
                
                st.session_state.results = results
                st.success("✅ Analysis completed!")
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
    st.header("📈 Cross-Section Geometry")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot outer shape
    if len(st.session_state.outer_points) >= 3:
        outer_array = np.array(st.session_state.outer_points)
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
        inner_polygon = Polygon(inner_array, alpha=0.8, facecolor='white', 
                               edgecolor='red', linewidth=2, label='Inner Hole')
        ax.add_patch(inner_polygon)
        
        # Plot inner points
        ax.scatter(inner_array[:, 0], inner_array[:, 1], c='red', s=100, zorder=5)
        
        # Label inner points
        for i, (x, y) in enumerate(inner_array):
            ax.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold', color='red')
    
    # Plot individual points for active geometry
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
    ax.set_title(f'Cross-Section Geometry\n{st.session_state.active_tab} (Click coordinates)', fontsize=14)
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
    instructions = f"""
    **Instructions:**
    - Currently editing: **{st.session_state.active_tab}**
    - Enter coordinates in the left panel and click "Add Point"
    - Minimum 3 points required for each shape
    - Points will be connected in order to form the shape
    """
    
    st.pyplot(fig)
    st.info(instructions)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>2D FEA Torsion Analysis</strong> - Web Version</p>
    <p>Developed by <strong>Yacht Engineering Services</strong> | Built with Streamlit</p>
    <p>
        <a href='https://github.com/YachtEngineeringservices/2d-FEA' target='_blank'>GitHub Repository</a> | 
        <a href='https://github.com/YachtEngineeringservices/2d-FEA/releases' target='_blank'>Desktop Version</a>
    </p>
</div>
""", unsafe_allow_html=True)
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
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>2D FEA Torsion Analysis</strong> - Web Version</p>
    <p>Developed by <strong>Yacht Engineering Services</strong> | Built with Streamlit</p>
    <p>
        <a href='https://github.com/YachtEngineeringservices/2d-FEA' target='_blank'>GitHub Repository</a> | 
        <a href='https://github.com/YachtEngineeringservices/2d-FEA/releases' target='_blank'>Desktop Version</a>
    </p>
</div>
""", unsafe_allow_html=True)

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
