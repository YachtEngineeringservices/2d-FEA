import numpy as np
import logging
import os

# Protected imports for FEniCS/PETSC
try:
    import dolfinx
    from dolfinx import fem, mesh, io
    from dolfinx.fem.petsc import LinearProblem
    from mpi4py import MPI
    import ufl
    FENICS_AVAILABLE = True
except ImportError:
    FENICS_AVAILABLE = False
    # Create dummy classes to prevent import errors
    MPI = None
    ufl = None

log = logging.getLogger(__name__)

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
    if not FENICS_AVAILABLE:
        raise ImportError("FEniCS is not available. Cannot perform torsion analysis.")
        
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

    # --- 5. Calculate Torsional Constant (J) ---
    # J = integral(psi) dA over the cross-section
    J_form = psi_h * ufl.dx
    J = fem.assemble_scalar(fem.form(J_form))
    log.info(f"Calculated Torsional Constant (J): {J:.4e} m^4")

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

    # Return the stress magnitude field for plotting instead of the stress function
    return J, k, theta, tau_max, tau_magnitude, V_mag
