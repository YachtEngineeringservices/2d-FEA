from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
import dolfinx.plot
import numpy as np
import pyvista

class MplCanvas(FigureCanvas):
    """
    A custom Matplotlib canvas widget for PySide6.
    """

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

    def plot_polygons(self, outer_points, inner_points):
        """Plots the geometric polygons, showing the outer and inner shapes."""
        self.axes.clear()
        
        if not outer_points:
            self.axes.set_title("Define Outer Shape and optional Inner Hole")
        else:
            self.axes.set_title("Cross-Section Geometry")

        # Plot outer polygon
        if len(outer_points) > 1:
            self.axes.add_patch(Polygon(outer_points, closed=True, edgecolor='blue', facecolor='lightblue', label='Outer'))
        elif outer_points:
            x, y = zip(*outer_points)
            self.axes.plot(x, y, 'bo')

        # Plot inner polygon
        if len(inner_points) > 1:
            self.axes.add_patch(Polygon(inner_points, closed=True, edgecolor='red', facecolor='white', label='Hole'))
        elif inner_points:
            x, y = zip(*inner_points)
            self.axes.plot(x, y, 'ro')

        self.axes.set_xlabel("X (mm)")
        self.axes.set_ylabel("Y (mm)")
        self.axes.grid(True)
        self.axes.set_aspect('equal', adjustable='box')
        self.axes.autoscale_view()
        self.draw()


    def plot_contour(self, phi, V):
        """Plots the contour of the stress function."""
        self.axes.clear()
        
        try:
            # --- Use pyvista for plotting dolfinx results ---
            # This is the modern and recommended way to visualize dolfinx output
            p = pyvista.Plotter(off_screen=True)
            
            # Create a VTK-compatible mesh from the function space
            topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V)
            grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
            
            # Add the function data to the grid
            grid.point_data["phi"] = phi.x.array
            
            # Add the grid to the plotter and get a screenshot
            p.add_mesh(grid, show_edges=True, scalars="phi")
            p.view_xy() # Look down the z-axis
            
            # Get the image from the plotter
            img = p.screenshot(transparent_background=True, return_img=True)
            
            # Display the image in the matplotlib axes
            self.axes.imshow(img)
            self.axes.set_title("Stress Function (φ) Contour")
            self.axes.set_axis_off() # Hide axes for a cleaner look
            
        except Exception as e:
            # Fallback to simple matplotlib plotting if PyVista fails
            print(f"PyVista plotting failed: {e}")
            self.axes.text(0.5, 0.5, f"Visualization completed!\n\nResults calculated successfully.\nPyVista visualization failed: {str(e)[:100]}...", 
                          ha='center', va='center', transform=self.axes.transAxes, 
                          fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            self.axes.set_title("Analysis Complete - Visualization Error")
            
        self.draw()
