import gmsh
import meshio
import logging
import os
import signal
import threading

log = logging.getLogger(__name__)

import gmsh
import meshio
import logging
import os
import signal
import threading
import multiprocessing
import subprocess
import sys

log = logging.getLogger(__name__)

def create_mesh_subprocess(outer_points, inner_points, mesh_size, output_dir):
    """Run GMSH mesh generation in a separate process to avoid signal conflicts."""
    
    # Create a temporary Python script to run GMSH
    script_content = f'''
import gmsh
import meshio
import os
import sys

def create_polygon(points, mesh_size):
    """Helper function to create a gmsh polygon from a list of points."""
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

def main():
    outer_points = {outer_points}
    inner_points = {inner_points}
    mesh_size = {mesh_size}
    output_dir = "{output_dir}"
    
    try:
        # Initialize GMSH in subprocess (no signal conflicts)
        gmsh.initialize()
        gmsh.model.add("torsion_section")
        
        # Create the outer polygon
        outer_loop = create_polygon(outer_points, mesh_size)
        
        # Create the inner polygon (hole) if it exists
        loops = [outer_loop]
        if inner_points:
            inner_loop = create_polygon(inner_points, mesh_size)
            loops.append(inner_loop)

        # Synchronize the geometry first
        gmsh.model.geo.synchronize()
        
        # Create the surface
        surface = gmsh.model.geo.addPlaneSurface(loops)
        
        # Synchronize again after creating the surface
        gmsh.model.geo.synchronize()

        # Add Physical Groups
        gmsh.model.addPhysicalGroup(2, [surface], 1)
        boundary_entities = gmsh.model.getBoundary([(2, surface)])
        boundary_lines = [ent[1] for ent in boundary_entities]
        gmsh.model.addPhysicalGroup(1, boundary_lines, 2)
        
        # Generate the 2D mesh
        gmsh.model.mesh.generate(2)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save mesh
        msh_filename = os.path.join(output_dir, "mesh.msh")
        gmsh.write(msh_filename)
        gmsh.finalize()
        
        # Convert to XDMF
        msh = meshio.read(msh_filename)
        triangle_cells = msh.get_cells_type("triangle")
        triangle_data = msh.get_cell_data("gmsh:physical", "triangle")
        
        domain_mesh = meshio.Mesh(
            points=msh.points,
            cells=[("triangle", triangle_cells)],
            cell_data={{"name_to_read": [triangle_data]}}
        )

        line_cells = msh.get_cells_type("line")
        line_data = msh.get_cell_data("gmsh:physical", "line")
        
        facet_mesh = meshio.Mesh(
            points=msh.points,
            cells=[("line", line_cells)],
            cell_data={{"name_to_read": [line_data]}}
        )

        meshio.write(os.path.join(output_dir, "domain.xdmf"), domain_mesh)
        meshio.write(os.path.join(output_dir, "facets.xdmf"), facet_mesh)
        
        # Clean up MSH file
        os.remove(msh_filename)
        
        print("SUCCESS")
        
    except Exception as e:
        print(f"ERROR: {{e}}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    # Write the script to a temporary file
    script_path = os.path.join(output_dir, "gmsh_subprocess.py")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Run the script in a subprocess
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, timeout=60)
        
        # Clean up the temporary script
        os.remove(script_path)
        
        if result.returncode == 0 and "SUCCESS" in result.stdout:
            return output_dir
        else:
            log.error(f"GMSH subprocess failed: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        log.error("GMSH subprocess timed out")
        return None
    except Exception as e:
        log.error(f"Failed to run GMSH subprocess: {e}")
        return None

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
    Creates a 2D mesh using subprocess to avoid signal conflicts with Streamlit.
    
    Args:
        outer_points (list): A list of (x, y) coordinates for the outer polygon.
        inner_points (list): A list of (x, y) coordinates for the inner polygon (hole).
        mesh_size (float): The desired characteristic length of the mesh elements.
        
    Returns:
        str: The path to the output directory.
    """
    log.info(f"Creating mesh using subprocess with {len(outer_points)} outer points, "
             f"{len(inner_points)} inner points, and mesh size {mesh_size}.")
    log.info(f"Outer points: {outer_points}")
    if inner_points:
        log.info(f"Inner points: {inner_points}")
    
    output_dir = "output"
    
    # Use subprocess approach to avoid signal conflicts
    result = create_mesh_subprocess(outer_points, inner_points, mesh_size, output_dir)
    
    if result:
        log.info("Mesh generation completed successfully using subprocess")
        return result
    else:
        log.error("Mesh generation failed")
        return None
