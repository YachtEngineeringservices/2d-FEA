import numpy as np
import logging
import os

log = logging.getLogger(__name__)

# Try to import DOLFINx components - fail gracefully if not available
try:
    import dolfinx
    from dolfinx import fem, mesh, io
    from dolfinx.fem.petsc import LinearProblem
    from mpi4py import MPI
    import ufl
    DOLFINX_AVAILABLE = True
    log.info("DOLFINx successfully imported")
except ImportError as e:
    DOLFINX_AVAILABLE = False
    log.warning(f"DOLFINx not available: {e}")
    # Define dummy classes/functions to prevent import errors
    class DummyDOLFINx:
        pass
    dolfinx = DummyDOLFINx()
    fem = DummyDOLFINx()
    mesh = DummyDOLFINx()
    io = DummyDOLFINx()
    LinearProblem = DummyDOLFINx()
    MPI = DummyDOLFINx()
    ufl = DummyDOLFINx()

def solve_torsion_3d_beam(mesh_dir, G, T, L, fixed_end='left'):
    """
    Solves 3D torsion problem for a beam with one fixed end and torque applied at the other.
    
    Args:
        mesh_dir (str): Path to the directory containing the XDMF mesh files.
        G (float): Shear modulus of the material (in Pascals).
        T (float): Applied torque (in N-m).
        L (float): Length of the beam (in meters).
        fixed_end (str): Which end is fixed - 'left' or 'right'
        
    Returns:
        tuple: Similar to solve_torsion but with 3D results
    """
    if not DOLFINX_AVAILABLE:
        raise RuntimeError("DOLFINx is required for 3D beam analysis but is not available.")
    
    # This would require creating a 3D mesh from the 2D cross-section
    # and implementing proper 3D boundary conditions
    
    # For now, return modified 2D results with end constraint effects
    # This is a simplified approach - proper 3D analysis would be more complex
    
    # Call the existing 2D solver
    J, k, theta_free, tau_max, tau_magnitude, V_mag, psi_h = solve_torsion(mesh_dir, G, T, L)
    
    # Apply end constraint correction factors
    # These are approximate - full 3D analysis would be needed for accuracy
    if fixed_end in ['left', 'right']:
        # Constrained warping increases stiffness and reduces twist
        # This is a simplified approximation
        constraint_factor = 1.2  # Approximate increase in stiffness
        k_constrained = k * constraint_factor
        theta_constrained = T / k_constrained
        
        log.info(f"Applied end constraint correction: k increased by {constraint_factor}x")
        return J, k_constrained, theta_constrained, tau_max, tau_magnitude, V_mag, psi_h
    else:
        return J, k, theta_free, tau_max, tau_magnitude, V_mag, psi_h

def solve_torsion(mesh_dir, G, T, L):
    """
    Solves the torsion problem for a given cross-section mesh.
    
    Args:
        mesh_dir (str): Path to the directory containing the XDMF mesh files.
        G (float): Shear modulus of the material (in Pascals).
        T (float): Applied torque (in N-m).
        L (float): Length of the beam (in meters).
        
    Returns:
        tuple: A tuple containing:
            - J (float): Torsional constant (m^4).
            - k (float): Torsional stiffness (Nm/rad).
            - theta (float): Angle of twist (radians).
            - tau_max (float): Maximum shear stress (Pascals).
            - tau_magnitude (dolfinx.fem.Function): The shear stress magnitude field.
            - V_mag (dolfinx.fem.FunctionSpace): The function space for the stress field.
    """
    comm = MPI.COMM_WORLD
    log.info("Starting torsion solver.")

    # --- 1. Read Mesh and Define Function Space ---
    domain_path = os.path.join(mesh_dir, "domain.xdmf")
    facets_path = os.path.join(mesh_dir, "facets.xdmf")
    log.info(f"Reading mesh from {domain_path} and {facets_path}")
    try:
        domain = io.XDMFFile(comm, domain_path, "r").read_mesh(name="Grid")
        
        # Ensure 1D entities (facets) and connectivity are created
        tdim = domain.topology.dim
        fdim = tdim - 1
        domain.topology.create_entities(fdim)
        domain.topology.create_connectivity(fdim, tdim)
        
        facet_tags = io.XDMFFile(comm, facets_path, "r").read_meshtags(domain, name="Grid")
    except Exception as e:
        log.error("Failed to read mesh files.", exc_info=True)
        raise

    V = fem.functionspace(domain, ("CG", 1))
    log.info(f"Function space created with {V.dofmap.index_map.size_global} degrees of freedom.")

    # --- 2. Define Boundary Conditions ---
    # The stress function phi is zero on the boundary of the cross-section.
    boundary_dofs = fem.locate_dofs_topological(V, facet_tags.dim, facet_tags.indices[facet_tags.values == 2])
    bc = fem.dirichletbc(fem.Constant(domain, 0.0), boundary_dofs, V)
    log.info(f"Dirichlet boundary condition applied to {len(boundary_dofs)} DOFs.")

    # --- 3. Define Variational Problem ---
    # The problem is governed by Poisson's equation: nabla^2(psi) = -2
    # We solve for the normalized stress function psi.
    psi = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.dot(ufl.grad(psi), ufl.grad(v)) * ufl.dx
    f = fem.Constant(domain, 2.0) # The source term is 2 for this formulation
    L_form = f * v * ufl.dx

    # --- 4. Solve the Linear Problem ---
    log.info("Setting up and solving the linear problem.")
    problem = LinearProblem(a, L_form, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    try:
        psi_h = problem.solve()
        log.info("Linear problem solved successfully.")
    except Exception as e:
        log.error("PETSc solver failed.", exc_info=True)
        raise

    # --- 5. Calculate Torsional Constant (J) and Cross-sectional Area ---
    # For the equation nabla^2(psi) = -2, the torsional constant is:
    # J = 2 * integral(psi) dA over the cross-section
    J_form = psi_h * ufl.dx
    J_integral = fem.assemble_scalar(fem.form(J_form))
    J_raw = 2.0 * J_integral  # Apply the factor of 2 from Saint-Venant theory
    
    # Calculate cross-sectional area for verification
    area_form = fem.Constant(domain, 1.0) * ufl.dx
    area = fem.assemble_scalar(fem.form(area_form))  # Keep area unscaled
    
    # DEBUG: Calculate PROPER area-weighted centroid (not nodal average!)
    x_coord = ufl.SpatialCoordinate(domain)[0]
    y_coord = ufl.SpatialCoordinate(domain)[1]
    
    # Area-weighted centroid calculation
    x_centroid_form = x_coord * ufl.dx
    y_centroid_form = y_coord * ufl.dx
    
    x_centroid = fem.assemble_scalar(fem.form(x_centroid_form)) / area
    y_centroid = fem.assemble_scalar(fem.form(y_centroid_form)) / area
    
    # Define coordinate functions relative to PROPER centroid
    x = ufl.SpatialCoordinate(domain)[0] - x_centroid
    y = ufl.SpatialCoordinate(domain)[1] - y_centroid
    
    # Calculate second moments of inertia about centroidal axes
    Ixx_form = y * y * ufl.dx  # Second moment about x-axis (y²dA)
    Iyy_form = x * x * ufl.dx  # Second moment about y-axis (x²dA)
    Ixy_form = x * y * ufl.dx  # Product moment of inertia
    
    Ixx = fem.assemble_scalar(fem.form(Ixx_form))  # m⁴
    Iyy = fem.assemble_scalar(fem.form(Iyy_form))  # m⁴
    Ixy = fem.assemble_scalar(fem.form(Ixy_form))  # m⁴
    
    # CORRECTED SCALING: Based on circle test, the 10× factor was too much
    # Circle test shows J is ~9.5× too large with 10× scaling
    # This suggests we need ~1.05× scaling instead of 10×, or possibly no scaling
    # Let's remove the scaling and see if the original J with factor of 2 is correct
    J = J_raw  # No additional scaling - just the factor of 2
    
    # DEBUG: Export mesh coordinates for analysis
    mesh_coords = V.tabulate_dof_coordinates()[:, :2]  # Get x,y coordinates
    x_min, x_max = np.min(mesh_coords[:, 0]), np.max(mesh_coords[:, 0])
    y_min, y_max = np.min(mesh_coords[:, 1]), np.max(mesh_coords[:, 1])
    
    # Export mesh coordinates to file for detailed analysis
    try:
        np.savetxt('output/mesh_coordinates.csv', mesh_coords, delimiter=',', 
                   header='X_meters,Y_meters', comments='')
        log.info(f"Mesh coordinates exported to output/mesh_coordinates.csv")
        log.info(f"Total mesh nodes: {len(mesh_coords)}")
    except Exception as e:
        log.warning(f"Could not export mesh coordinates: {e}")
    
    log.info(f"DEBUG: Mesh coordinate ranges:")
    log.info(f"  X: [{x_min:.6f}, {x_max:.6f}] (range: {x_max-x_min:.6f})")
    log.info(f"  Y: [{y_min:.6f}, {y_max:.6f}] (range: {y_max-y_min:.6f})")
    log.info(f"Area: {area:.6f} m² = {area*1e6:.1f} mm²")
    log.info(f"Area-weighted Centroid: ({x_centroid:.6f}, {y_centroid:.6f}) m")
    log.info(f"Rhino Centroid Reference: (-0.0000359, 0.535466) m")
    
    # DEBUG: Calculate some intermediate values for Ixx debugging
    # Test the coordinate system by calculating simple integrals
    x_coord_check = ufl.SpatialCoordinate(domain)[0]
    y_coord_check = ufl.SpatialCoordinate(domain)[1]
    
    # Raw coordinate integrals (before centering)
    raw_x_integral = fem.assemble_scalar(fem.form(x_coord_check * ufl.dx))
    raw_y_integral = fem.assemble_scalar(fem.form(y_coord_check * ufl.dx))
    raw_y2_integral = fem.assemble_scalar(fem.form(y_coord_check * y_coord_check * ufl.dx))
    
    # Mesh quality analysis
    y_coords = mesh_coords[:, 1]
    y_distribution = {
        'y_min': np.min(y_coords),
        'y_max': np.max(y_coords), 
        'y_mean': np.mean(y_coords),
        'y_std': np.std(y_coords),
        'nodes_top_quarter': np.sum(y_coords > (y_centroid + (y_max - y_centroid) * 0.5)),
        'nodes_bottom_quarter': np.sum(y_coords < (y_centroid - (y_centroid - y_min) * 0.5)),
        'total_nodes': len(y_coords)
    }
    
    log.info(f"DEBUG: Raw coordinate integrals:")
    log.info(f"  ∫x dA = {raw_x_integral:.6e} m³ (should ≈ x_centroid × area)")
    log.info(f"  ∫y dA = {raw_y_integral:.6e} m³ (should ≈ y_centroid × area)")
    log.info(f"  ∫y² dA = {raw_y2_integral:.6e} m⁴")
    log.info(f"Expected: x_cent×area = {x_centroid*area:.6e}, y_cent×area = {y_centroid*area:.6e}")
    
    log.info(f"DEBUG: Y-coordinate mesh distribution:")
    log.info(f"  Nodes in top quarter: {y_distribution['nodes_top_quarter']} ({y_distribution['nodes_top_quarter']/y_distribution['total_nodes']*100:.1f}%)")
    log.info(f"  Nodes in bottom quarter: {y_distribution['nodes_bottom_quarter']} ({y_distribution['nodes_bottom_quarter']/y_distribution['total_nodes']*100:.1f}%)")
    log.info(f"  Y-coordinate std dev: {y_distribution['y_std']:.6f} m")
    
    log.info(f"Second Moments of Inertia about area-weighted centroid:")
    log.info(f"  Ixx: {Ixx:.6e} m⁴ = {Ixx*1e12:.1f} mm⁴")
    log.info(f"  Iyy: {Iyy:.6e} m⁴ = {Iyy*1e12:.1f} mm⁴")
    log.info(f"  Ixy: {Ixy:.6e} m⁴ = {Ixy*1e12:.1f} mm⁴")
    
    # Additional debugging: check parallel axis theorem
    raw_ixx_about_origin = raw_y2_integral
    parallel_axis_correction = area * y_centroid * y_centroid
    ixx_calculated_manually = raw_ixx_about_origin - parallel_axis_correction
    
    log.info(f"DEBUG: Parallel axis theorem check:")
    log.info(f"  Ixx about origin: {raw_ixx_about_origin:.6e} m⁴")
    log.info(f"  Parallel axis term: {parallel_axis_correction:.6e} m⁴")
    log.info(f"  Ixx manual calc: {ixx_calculated_manually:.6e} m⁴")
    log.info(f"  FEA Ixx: {Ixx:.6e} m⁴")
    log.info(f"  Difference: {abs(Ixx - ixx_calculated_manually):.2e} m⁴")
    
    log.info(f"Rhino Reference - Ixx: 7.011e9 mm⁴, Iyy: 4.461e9 mm⁴")
    log.info(f"J = 2*integral: {J:.6e} m⁴ (no additional scaling)")
    log.info(f"Calculated cross-sectional area: {area:.6f} m²")
    log.info(f"Calculated integral(psi): {J_integral:.4e}")
    log.info(f"Calculated Torsional Constant (J = 2*integral): {J:.4e} m^4")

    # --- 6. Calculate Torsional Stiffness (k) and Angle of Twist (theta) ---
    k = (G * J) / L  # Torsional stiffness
    theta = T / k    # Angle of twist in radians
    log.info(f"Calculated Stiffness (k): {k:.4e} Nm/rad, Angle of Twist (theta): {theta:.4e} rad")

    # --- 7. Calculate Shear Stress (tau) ---
    # The actual stress function is phi = G * (theta/L) * psi_h
    # Shear stress components are tau_zx = d(phi)/dy and tau_zy = -d(phi)/dx
    # tau = sqrt(tau_zx^2 + tau_zy^2)
    
    # We need a different function space for the gradients (Discontinuous Galerkin)
    W = fem.functionspace(domain, ("DG", 0, (domain.geometry.dim,)))
    grad_psi = ufl.grad(psi_h)
    
    # The expression for the stress vector (tau_zy, tau_zx, 0)
    # Note: twist per unit length is theta/L. We add a 0 z-component to match the 3D geometry space.
    stress_expr = fem.Expression(ufl.as_vector((-grad_psi[1], grad_psi[0], 0.0)) * (G * (theta/L)), W.element.interpolation_points())
    stress_field = fem.Function(W)
    stress_field.interpolate(stress_expr)

    # Calculate the magnitude of the stress vector at each point
    V_mag = fem.functionspace(domain, ("DG", 0))
    tau_mag_expr = fem.Expression(ufl.sqrt(stress_field[0]**2 + stress_field[1]**2), V_mag.element.interpolation_points())
    tau_magnitude = fem.Function(V_mag)
    tau_magnitude.interpolate(tau_mag_expr)

    # Find the maximum stress
    # Accessing the underlying array and finding the max value
    tau_max = np.max(tau_magnitude.x.array)
    log.info(f"Calculated Maximum Shear Stress (tau_max): {tau_max:.2f} Pa")

    # --- 8. Clean up temporary files ---
    # Remove mesh and output files to keep repository clean
    # Keep mesh_coordinates.csv for debugging
    try:
        cleanup_files = [
            os.path.join(mesh_dir, "domain.h5"),
            os.path.join(mesh_dir, "domain.xdmf"),
            os.path.join(mesh_dir, "facets.h5"),
            os.path.join(mesh_dir, "facets.xdmf"),
            os.path.join(mesh_dir, "mesh.msh")
        ]
        
        for file_path in cleanup_files:
            if os.path.exists(file_path):
                os.remove(file_path)
                log.debug(f"Cleaned up temporary file: {file_path}")
        
        log.info("Temporary mesh files cleaned up successfully (mesh_coordinates.csv preserved)")
    except Exception as e:
        log.warning(f"Failed to clean up some temporary files: {e}")

    # Return the stress magnitude field for plotting, and the warping function psi_h for deflection
    return J, k, theta, tau_max, tau_magnitude, V_mag, psi_h, area
