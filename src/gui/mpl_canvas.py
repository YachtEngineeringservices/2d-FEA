from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from matplotlib.widgets import Cursor
import numpy as np
import pyvista
try:
    from scipy.interpolate import griddata
except ImportError:
    griddata = None

class MplCanvas(FigureCanvas):
    """
    A custom Matplotlib canvas widget for PySide6 with zoom and pan capabilities.
    """

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Enable zoom and pan
        self.fig.canvas.toolbar_visible = False  # Hide default toolbar
        
        # Add zoom and pan functionality
        self.setup_zoom_pan()
        
        # Store original limits for reset
        self.original_xlim = None
        self.original_ylim = None
        
        # Store current geometry for hole masking
        self.current_inner_points = []
        
    def setup_zoom_pan(self):
        """Setup mouse wheel zoom and click-drag pan functionality."""
        self.mpl_connect('scroll_event', self.on_scroll)
        self.mpl_connect('button_press_event', self.on_press)
        self.mpl_connect('button_release_event', self.on_release)
        self.mpl_connect('motion_notify_event', self.on_motion)
        
        # Pan state
        self.pan_active = False
        self.pan_start = None
        
    def on_scroll(self, event):
        """Handle mouse wheel zoom."""
        if event.inaxes != self.axes:
            return
            
        # Zoom factor
        zoom_factor = 1.1 if event.button == 'up' else 1/1.1
        
        # Get current axes limits
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        
        # Calculate new limits centered on cursor
        xdata, ydata = event.xdata, event.ydata
        
        # Calculate zoom
        x_left = xdata - (xdata - xlim[0]) * zoom_factor
        x_right = xdata + (xlim[1] - xdata) * zoom_factor
        y_bottom = ydata - (ydata - ylim[0]) * zoom_factor
        y_top = ydata + (ylim[1] - ydata) * zoom_factor
        
        self.axes.set_xlim([x_left, x_right])
        self.axes.set_ylim([y_bottom, y_top])
        self.draw()
        
    def on_press(self, event):
        """Handle mouse press for pan start."""
        if event.inaxes != self.axes or event.button != 3:  # Right click only
            return
        self.pan_active = True
        self.pan_start = (event.xdata, event.ydata)
        
    def on_release(self, event):
        """Handle mouse release for pan end."""
        self.pan_active = False
        self.pan_start = None
        
    def on_motion(self, event):
        """Handle mouse motion for panning."""
        if not self.pan_active or not self.pan_start or event.inaxes != self.axes:
            return
            
        # Calculate pan delta
        dx = self.pan_start[0] - event.xdata
        dy = self.pan_start[1] - event.ydata
        
        # Apply pan
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        self.axes.set_xlim([xlim[0] + dx, xlim[1] + dx])
        self.axes.set_ylim([ylim[0] + dy, ylim[1] + dy])
        self.draw()
        
    def _point_in_polygon(self, x, y, poly):
        """Check if point (x, y) is inside polygon using ray casting algorithm."""
        if len(poly) < 3:
            return False  # Need at least 3 points for a polygon
        
        n = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(1, n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    
    def _create_hole_polygon(self, inner_points_mm):
        """Create a proper closed polygon from inner points."""
        if len(inner_points_mm) < 3:
            return []
        
        # Convert to numpy array for easier manipulation
        points = np.array(inner_points_mm)
        
        # Calculate centroid
        centroid = np.mean(points, axis=0)
        
        # Calculate angles from centroid to each point
        angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
        
        # Sort points by angle to create a proper polygon
        sorted_indices = np.argsort(angles)
        sorted_points = points[sorted_indices]
        
        print(f"Debug: Original points: {inner_points_mm}")
        print(f"Debug: Centroid: {centroid}")
        print(f"Debug: Sorted polygon: {sorted_points.tolist()}")
        
        return sorted_points.tolist()
    
    def _mask_holes(self, xi, yi, zi, inner_points_list):
        """Mask areas inside holes (inner polygons) by setting them to NaN."""
        # Convert inner points to match the plotting coordinate system (mm)
        # The inner_points_list is in meters, but xi/yi are in mm
        inner_points_plot = [(x * 1000, y * 1000) for x, y in inner_points_list]
        
        print(f"Debug: Masking holes with {len(inner_points_plot)} inner points")
        print(f"Debug: Inner points (mm): {inner_points_plot}")
        print(f"Debug: Grid range: X={xi.min():.1f}-{xi.max():.1f}, Y={yi.min():.1f}-{yi.max():.1f}")
        
        # Create mask for points inside the hole
        mask = np.zeros_like(zi, dtype=bool)
        points_masked = 0
        
        for i in range(xi.shape[0]):
            for j in range(xi.shape[1]):
                if self._point_in_polygon(xi[i, j], yi[i, j], inner_points_plot):
                    mask[i, j] = True
                    points_masked += 1
        
        print(f"Debug: Masked {points_masked} points out of {mask.size} total grid points")
        
        # Set masked areas to NaN so they appear empty
        zi[mask] = np.nan
        return zi

    def reset_view(self):
        """Reset the view to show all geometry."""
        if self.original_xlim and self.original_ylim:
            self.axes.set_xlim(self.original_xlim)
            self.axes.set_ylim(self.original_ylim)
        else:
            self.axes.autoscale_view()
        self.draw()

    def plot_polygons(self, outer_points, inner_points, selected_point_index=-1, active_is_outer=True):
        """Plots the geometric polygons, showing the outer and inner shapes with optional point highlighting."""
        self.axes.clear()
        
        # Store current points for hole masking
        self.current_inner_points = inner_points if inner_points else []
        
        if not outer_points:
            self.axes.set_title("Define Outer Shape and optional Inner Hole")
        else:
            self.axes.set_title("Cross-Section Geometry")

        # Plot outer polygon
        if len(outer_points) > 1:
            self.axes.add_patch(Polygon(outer_points, closed=True, edgecolor='blue', facecolor='lightblue', alpha=0.3, label='Outer Shape'))
            
        # Plot outer points
        if outer_points:
            x, y = zip(*outer_points)
            self.axes.plot(x, y, 'bo', markersize=8, picker=True, pickradius=10, label='Outer Points')
            
            # Highlight selected point if it's in the outer points and outer is active
            if selected_point_index >= 0 and active_is_outer and selected_point_index < len(outer_points):
                sel_x, sel_y = outer_points[selected_point_index]
                self.axes.plot(sel_x, sel_y, 'yo', markersize=12, markeredgecolor='orange', markeredgewidth=2)
                
            # Add point numbers
            for i, (px, py) in enumerate(outer_points):
                self.axes.annotate(f'{i+1}', (px, py), xytext=(5, 5), textcoords='offset points', 
                                 fontsize=12, color='blue', weight='bold')

        # Plot inner polygon
        if len(inner_points) > 1:
            self.axes.add_patch(Polygon(inner_points, closed=True, edgecolor='red', facecolor='white', label='Hole'))
            
        # Plot inner points
        if inner_points:
            x, y = zip(*inner_points)
            self.axes.plot(x, y, 'ro', markersize=8, picker=True, pickradius=10, label='Inner Points')
            
            # Highlight selected point if it's in the inner points and inner is active
            if selected_point_index >= 0 and not active_is_outer and selected_point_index < len(inner_points):
                sel_x, sel_y = inner_points[selected_point_index]
                self.axes.plot(sel_x, sel_y, 'yo', markersize=12, markeredgecolor='orange', markeredgewidth=2)
                
            # Add point numbers
            for i, (px, py) in enumerate(inner_points):
                self.axes.annotate(f'{i+1}', (px, py), xytext=(5, 5), textcoords='offset points', 
                                 fontsize=12, color='red', weight='bold')

        # Set equal aspect ratio and configure the plot
        self.axes.set_aspect('equal', adjustable='box')
        self.axes.grid(True, alpha=0.3)
        if outer_points or inner_points:
            self.axes.legend(loc='upper right')

        # Add instructions
        if outer_points or inner_points:
            instruction_text = "Mouse: Wheel=Zoom, Right-click+drag=Pan, Left-click=Add Point, Click point=Select"
            self.axes.text(0.02, 0.98, instruction_text, transform=self.axes.transAxes, 
                         fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                         facecolor="lightyellow", alpha=0.8))
        
        self.draw()


    def plot_contour(self, tau_magnitude, V_mag, inner_points=None):
        """Plots the contour of the shear stress magnitude with proper hole masking."""
        self.axes.clear()
        
        # Store inner points for hole masking
        if inner_points is not None:
            self.current_inner_points = inner_points
        elif not hasattr(self, 'current_inner_points'):
            self.current_inner_points = []

        try:
            # Get mesh coordinates and stress values
            coords = V_mag.tabulate_dof_coordinates()
            stress_values = tau_magnitude.x.array / 1e6  # Convert to MPa
            
            print(f"Debug: Using triangulation approach with hole masking")
            print(f"Debug: {len(coords)} coordinates, {len(stress_values)} stress values")
            print(f"Debug: Inner points: {len(self.current_inner_points) if self.current_inner_points else 0}")
            
            # Create matplotlib triangulation
            import matplotlib.tri as tri
            x = coords[:, 0] * 1000  # Convert to mm
            y = coords[:, 1] * 1000  # Convert to mm
            
            # Create triangulation
            triang = tri.Triangulation(x, y)
            
            # If we have holes, mask triangles that are inside holes
            if self.current_inner_points:
                print("Debug: Masking triangles inside holes...")
                # Convert inner points to mm - but check if they're already in mm or meters
                # The inner_points from DOLFINx are in meters, but we need mm for plotting
                inner_points_mm = []
                for px, py in self.current_inner_points:
                    # Check if points seem to be in meters (small values) or already in mm
                    if abs(px) < 1.0 and abs(py) < 1.0:
                        # Points are in meters, convert to mm
                        inner_points_mm.append((px * 1000, py * 1000))
                    else:
                        # Points are already in mm scale
                        inner_points_mm.append((px, py))
                
                print(f"Debug: Input inner points: {self.current_inner_points}")
                print(f"Debug: Converted to mm: {inner_points_mm}")
                print(f"Debug: Mesh coordinate range: X={x.min():.1f}-{x.max():.1f}, Y={y.min():.1f}-{y.max():.1f}")
                
                # Create proper closed polygon from the points
                hole_polygon = self._create_hole_polygon(inner_points_mm)
                
                if hole_polygon:
                    # Calculate triangle centers
                    tri_centers_x = x[triang.triangles].mean(axis=1)
                    tri_centers_y = y[triang.triangles].mean(axis=1)
                    
                    # Debug: Show some triangle centers
                    print(f"Debug: Sample triangle centers: {list(zip(tri_centers_x[:5], tri_centers_y[:5]))}")
                    
                    # Mask triangles whose centers are inside holes
                    mask = np.zeros(len(triang.triangles), dtype=bool)
                    masked_count = 0
                    for i, (cx, cy) in enumerate(zip(tri_centers_x, tri_centers_y)):
                        if self._point_in_polygon(cx, cy, hole_polygon):
                            mask[i] = True
                            masked_count += 1
                            if masked_count <= 5:  # Show first few masked triangles
                                print(f"Debug: Masking triangle {i} at center ({cx:.1f}, {cy:.1f})")
                    
                    triang.set_mask(mask)
                    print(f"Debug: Masked {np.sum(mask)} triangles out of {len(mask)}")
                else:
                    print("Debug: Could not create hole polygon from points")
            
            # Create smooth filled contour plot
            levels = np.linspace(stress_values.min(), stress_values.max(), 50)
            contour = self.axes.tricontourf(triang, stress_values, levels=levels, cmap='jet', extend='both')
            
            # Add contour lines for better definition
            contour_lines = self.axes.tricontour(triang, stress_values, levels=levels[::5], colors='black', alpha=0.3, linewidths=0.5)
            
            # Add colorbar with better formatting
            cbar = self.fig.colorbar(contour, ax=self.axes, shrink=0.8, pad=0.02)
            try:
                import matplotlib.ticker as ticker
                cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
            except ImportError:
                pass
            
            cbar.set_label('Shear Stress (MPa)', rotation=270, labelpad=20, fontsize=14, fontweight='bold')
            
            self.axes.set_title("Shear Stress Distribution", fontsize=16, fontweight='bold')
            self.axes.set_xlabel("X (mm)", fontsize=14)
            self.axes.set_ylabel("Y (mm)", fontsize=14)
            self.axes.set_aspect('equal', adjustable='box')
            
            # Add stress info
            max_stress = stress_values.max()
            self.axes.text(0.02, 0.98, f"Max Stress: {max_stress:.1f} MPa", 
                          transform=self.axes.transAxes, fontsize=14, 
                          verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            
        except Exception as e:
            print(f"Triangulation plotting failed: {e}")
            import traceback
            traceback.print_exc()
            # Ultimate fallback - just show text with results
            max_stress = tau_magnitude.x.array.max() / 1e6
            self.axes.text(0.5, 0.5, f"Visualization Error\nAnalysis completed successfully\nMax stress: {max_stress:.1f} MPa", 
                          ha='center', va='center', transform=self.axes.transAxes, 
                          fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        self.draw()
