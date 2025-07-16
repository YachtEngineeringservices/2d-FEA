import gmsh
import meshio
import logging
import os

log = logging.getLogger(__name__)

def validate_polygon(points):
    """Validate if points form a simple (non-self-intersecting) polygon."""
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    def intersect(A, B, C, D):
        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

    n = len(points)
    for i in range(n):
        for j in range(i+2, n):
            if j == n-1 and i == 0:  # Don't check last edge with first edge
                continue
            edge1 = (points[i], points[(i+1) % n])
            edge2 = (points[j], points[(j+1) % n])
            if intersect(edge1[0], edge1[1], edge2[0], edge2[1]):
                return False, f"Edges {i}-{(i+1)%n} and {j}-{(j+1)%n} intersect"
    return True, "Valid polygon"

def create_polygon(points, mesh_size):
    """Helper function to create a gmsh polygon from a list of points."""
    # Validate the polygon first
    is_valid, message = validate_polygon(points)
    if not is_valid:
        log.warning(f"Polygon validation failed: {message}")
        log.warning(f"Points: {points}")
        # Continue anyway, but let user know
    
    gmsh_points = []
    for p in points:
        gmsh_points.append(gmsh.model.geo.addPoint(p[0], p[1], 0, meshSize=mesh_size))
    
    lines = []
    for i in range(len(gmsh_points)):
        p1 = gmsh_points[i]
        p2 = gmsh_points[(i + 1) % len(gmsh_points)]
        lines.append(gmsh.model.geo.addLine(p1, p2))
        
    loop = gmsh.model.geo.addCurveLoop(lines)
    return loop

def create_mesh(outer_points, inner_points, mesh_size):
    """
    Creates a 2D mesh, potentially with a hole.
    
    Args:
        outer_points (list): A list of (x, y) coordinates for the outer polygon.
        inner_points (list): A list of (x, y) coordinates for the inner polygon (hole).
        mesh_size (float): The desired characteristic length of the mesh elements.
        
    Returns:
        str: The path to the output directory.
    """
    log.info(f"Initializing gmsh for meshing with {len(outer_points)} outer points, "
             f"{len(inner_points)} inner points, and mesh size {mesh_size}.")
    log.info(f"Outer points: {outer_points}")
    if inner_points:
        log.info(f"Inner points: {inner_points}")
        
    gmsh.initialize()
    gmsh.model.add("torsion_section")

    try:
        # Create the outer polygon
        outer_loop = create_polygon(outer_points, mesh_size)
        
        # Create the inner polygon (hole) if it exists
        loops = [outer_loop]
        if inner_points:
            inner_loop = create_polygon(inner_points, mesh_size)
            loops.append(inner_loop)

        # Synchronize the geometry first
        gmsh.model.geo.synchronize()
        
        # Create the surface, subtracting the inner loop if it exists
        surface = gmsh.model.geo.addPlaneSurface(loops)
        
        # Synchronize again after creating the surface
        gmsh.model.geo.synchronize()

        # --- Add Physical Groups ---
        # This is the crucial step to label the domain and boundary
        gmsh.model.addPhysicalGroup(2, [surface], 1) # Tag for the surface (domain)
        
        # Get all boundary lines. Gmsh automatically knows the boundaries of the surface.
        boundary_entities = gmsh.model.getBoundary([(2, surface)])
        boundary_lines = [ent[1] for ent in boundary_entities]
        gmsh.model.addPhysicalGroup(1, boundary_lines, 2) # Tag for all boundary lines
        
        # Generate the 2D mesh
        gmsh.model.mesh.generate(2)
        
    except Exception as e:
        log.error(f"GMSH geometry/mesh generation failed: {e}")
        # Save the geometry for debugging
        try:
            gmsh.write("debug_geometry.geo_unrolled")
            log.info("Debug geometry saved to debug_geometry.geo_unrolled")
        except:
            pass
        gmsh.finalize()
        raise

    # --- Save the mesh to a file ---
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        log.info(f"Created output directory: {output_dir}")

    msh_filename = os.path.join(output_dir, "mesh.msh")
    
    log.info(f"Writing mesh to {msh_filename}")
    gmsh.write(msh_filename)
    gmsh.finalize()

    # --- Convert MSH to XDMF for dolfinx ---
    log.info(f"Converting {msh_filename} to XDMF format using meshio.")
    try:
        msh = meshio.read(msh_filename)

        # Extract cells and data for triangles (domain)
        triangle_cells = msh.get_cells_type("triangle")
        triangle_data = msh.get_cell_data("gmsh:physical", "triangle")
        
        # Create a new meshio.Mesh object for the domain, keeping only the domain cells
        domain_mesh = meshio.Mesh(
            points=msh.points,
            cells=[("triangle", triangle_cells)],
            cell_data={"name_to_read": [triangle_data]}
        )

        # Extract cells and data for lines (facets/boundary)
        line_cells = msh.get_cells_type("line")
        line_data = msh.get_cell_data("gmsh:physical", "line")
        
        # Create a new meshio.Mesh object for the facets
        facet_mesh = meshio.Mesh(
            points=msh.points,
            cells=[("line", line_cells)],
            cell_data={"name_to_read": [line_data]}
        )

        # Write the domain and facet meshes to XDMF files
        meshio.write(os.path.join(output_dir, "domain.xdmf"), domain_mesh)
        meshio.write(os.path.join(output_dir, "facets.xdmf"), facet_mesh)
        log.info("Successfully created domain.xdmf and facets.xdmf")

    except Exception as e:
        log.error("Failed to convert MSH to XDMF.", exc_info=True)
        raise

    return output_dir
