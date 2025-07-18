"""
FEA package for 2D torsional analysis.

Direct imports for cloud deployment - requires DOLFINx
"""

# Direct imports for cloud deployment
from . import meshing
from . import solver

# Set availability flags
MESHING_AVAILABLE = True
SOLVER_AVAILABLE = "full"

__all__ = ['solver', 'meshing', 'SOLVER_AVAILABLE', 'MESHING_AVAILABLE']
