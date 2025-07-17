"""
FEA package for 2D torsional analysis.

Provides conditional imports for cross-platform compatibility:
- Full solver with DOLFINx for Linux/advanced analysis
- Simplified solver for Windows compatibility
"""

# Conditional imports for cross-platform compatibility
try:
    # Try to import full DOLFINx-based solver
    from . import solver
    SOLVER_AVAILABLE = "full"
    print("DOLFINx solver available")
except ImportError as dolfinx_error:
    try:
        # Fall back to Windows-compatible simplified solver
        from . import solver_windows as solver
        SOLVER_AVAILABLE = "simplified"
        print(f"Using simplified solver (DOLFINx not available: {dolfinx_error})")
    except ImportError as solver_error:
        # No solver available
        print(f"Warning: No solver available. DOLFINx error: {dolfinx_error}, Simplified solver error: {solver_error}")
        solver = None
        SOLVER_AVAILABLE = "none"

# Always import meshing (no special dependencies)
try:
    from . import meshing
    MESHING_AVAILABLE = True
except ImportError as mesh_error:
    print(f"Warning: Meshing not available: {mesh_error}")
    meshing = None
    MESHING_AVAILABLE = False

__all__ = ['solver', 'meshing', 'SOLVER_AVAILABLE', 'MESHING_AVAILABLE']
