#!/usr/bin/env python3
"""
Mixed Section Analysis for thin-walled sections with both open and closed parts
"""

import sys
import os
sys.path.append('/home/adminlinux/2d FEA/src')

import numpy as np
from typing import List, Tuple, Dict, Any
from fea.multi_method_analysis import MultiMethodAnalysis

class MixedSectionAnalysis:
    """
    Analyzes thin-walled sections that have both open and closed portions
    Common in ship hulls, aircraft fuselages, and automotive structures
    """
    
    def __init__(self, outer_points: List[Tuple[float, float]], 
                 inner_points: List[Tuple[float, float]] = None,
                 wall_thickness: float = None):
        """
        Initialize mixed section analysis
        
        Args:
            outer_points: Outer boundary points (m)
            inner_points: Inner boundary points (m) - defines closed cavity
            wall_thickness: Wall thickness (m) - if None, will estimate
        """
        self.outer_points = outer_points
        self.inner_points = inner_points or []
        self.wall_thickness = wall_thickness
        
    def analyze_mixed_section(self, G: float, T: float, L: float) -> Dict[str, Any]:
        """
        Analyze mixed open/closed thin-walled section
        
        Returns:
            Complete analysis with decomposed contributions and stress data
        """
        results = {
            'geometry': self._analyze_geometry(),
            'closed_contribution': self._analyze_closed_part(),
            'open_contribution': self._analyze_open_parts(),
            'stress_analysis': self._analyze_stress_distribution(T),
            'total': {},
            'method_notes': []
        }
        
        # Combine contributions
        J_closed = results['closed_contribution'].get('J', 0) or 0
        J_open = results['open_contribution'].get('J', 0) or 0
        J_total = J_closed + J_open
        
        # Calculate total performance
        if J_total > 0:
            k_total = G * J_total / L
            theta_rad = T / k_total
            theta_deg = np.rad2deg(theta_rad)
            
            results['total'] = {
                'J': J_total,
                'J_closed': J_closed,
                'J_open': J_open,
                'closed_fraction': J_closed / J_total if J_total > 0 else 0,
                'open_fraction': J_open / J_total if J_total > 0 else 0,
                'k': k_total,
                'theta_rad': theta_rad,
                'theta_deg': theta_deg,
                'method': 'Mixed section (Bredt + open thin-wall)'
            }
        
        return results
    
    def _analyze_geometry(self) -> Dict[str, Any]:
        """Analyze basic geometric properties"""
        
        # Use the existing multi-method analyzer for basic geometry
        multi_analyzer = MultiMethodAnalysis(self.outer_points, self.inner_points)
        geometry = multi_analyzer._analyze_geometry()
        
        # Add mixed-section specific geometry analysis
        if self.inner_points:
            # Closed cavity exists
            closed_area = abs(self._polygon_area_signed(self.inner_points))
            open_perimeter = self._calculate_open_perimeter()
        else:
            # Fully open section
            closed_area = 0
            open_perimeter = self._polygon_perimeter(self.outer_points)
        
        geometry.update({
            'closed_cavity_area': closed_area,
            'open_perimeter': open_perimeter,
            'section_type': self._classify_section_type(),
            'wall_thickness_estimated': self.wall_thickness or geometry.get('wall_thickness', 0)
        })
        
        return geometry
    
    def _analyze_closed_part(self) -> Dict[str, Any]:
        """Analyze the closed cavity portion using Bredt's method"""
        
        if not self.inner_points:
            return {
                'J': 0,
                'method': 'No closed cavity',
                'note': 'Section is fully open'
            }
        
        # Use Bredt's formula for the closed cavity
        # J = 4A²/∮(ds/t)
        
        # Enclosed area (use inner boundary as the cavity)
        enclosed_area = abs(self._polygon_area_signed(self.inner_points))
        
        # Wall thickness
        t = self.wall_thickness or self._estimate_wall_thickness()
        
        if t <= 0:
            return {
                'J': None,
                'method': 'Bredt (failed - no wall thickness)',
                'note': 'Cannot determine wall thickness'
            }
        
        # Perimeter of the closed cavity (inner boundary)
        closed_perimeter = self._polygon_perimeter(self.inner_points)
        
        # Bredt's formula
        if closed_perimeter > 0:
            J_closed = 4 * (enclosed_area ** 2) / (closed_perimeter / t)
        else:
            J_closed = 0
        
        return {
            'J': J_closed,
            'method': 'Bredt formula (closed cavity)',
            'enclosed_area': enclosed_area,
            'closed_perimeter': closed_perimeter,
            'wall_thickness': t,
            'note': f'Closed cavity contributes J = {J_closed:.4e} m⁴'
        }
    
    def _analyze_open_parts(self) -> Dict[str, Any]:
        """Analyze the open portions using thin-walled open section theory"""
        
        # Calculate the "open" perimeter - parts not enclosing cavity
        open_perimeter = self._calculate_open_perimeter()
        
        if open_perimeter <= 0:
            return {
                'J': 0,
                'method': 'No open parts',
                'note': 'Section is fully closed'
            }
        
        # Wall thickness
        t = self.wall_thickness or self._estimate_wall_thickness()
        
        if t <= 0:
            return {
                'J': None,
                'method': 'Open thin-wall (failed - no wall thickness)',
                'note': 'Cannot determine wall thickness'
            }
        
        # Open section formula: J = (1/3) * ∑(b * t³)
        # For uniform thickness: J = (1/3) * t³ * total_open_length
        J_open = (1/3) * open_perimeter * (t ** 3)
        
        return {
            'J': J_open,
            'method': 'Open thin-walled theory',
            'open_perimeter': open_perimeter,
            'wall_thickness': t,
            'note': f'Open parts contribute J = {J_open:.4e} m⁴'
        }
    
    def _calculate_open_perimeter(self) -> float:
        """
        Calculate the perimeter of open parts
        This is the outer perimeter minus the parts that form the closed cavity
        """
        
        total_outer_perimeter = self._polygon_perimeter(self.outer_points)
        
        if not self.inner_points:
            # Fully open section
            return total_outer_perimeter
        
        # For mixed sections, estimate the open perimeter
        # This is approximate - exact calculation would require topology analysis
        
        # Method: Assume the closed cavity reduces the "effective open length"
        inner_perimeter = self._polygon_perimeter(self.inner_points)
        
        # Rough approximation: open perimeter = outer - inner
        # This assumes the closed part "uses up" some of the outer perimeter
        open_perimeter = max(0, total_outer_perimeter - inner_perimeter)
        
        return open_perimeter
    
    def _analyze_stress_distribution(self, T: float) -> Dict[str, Any]:
        """
        Analyze stress distribution for visualization
        
        For closed sections: τ = q/t where q = T/(2A) is shear flow
        For open sections: τ varies linearly from zero at free edges
        
        Returns stress data for plotting
        """
        
        stress_data = {
            'closed_stress': [],
            'open_stress': [],
            'max_stress_location': None,
            'max_stress_value': 0
        }
        
        # Wall thickness
        t = self.wall_thickness or self._estimate_wall_thickness()
        
        if t <= 0:
            return stress_data
        
        # Closed section stress (if exists)
        if self.inner_points:
            enclosed_area = abs(self._polygon_area_signed(self.inner_points))
            if enclosed_area > 0:
                # Shear flow: q = T/(2A)
                q = T / (2 * enclosed_area)
                
                # Shear stress: τ = q/t (constant around closed loop)
                tau_closed = q / t
                
                # Create stress points around inner boundary
                for i, point in enumerate(self.inner_points):
                    stress_data['closed_stress'].append({
                        'x': point[0],
                        'y': point[1], 
                        'tau': tau_closed,
                        'type': 'closed',
                        'segment': i
                    })
                
                # Track maximum stress
                if tau_closed > stress_data['max_stress_value']:
                    stress_data['max_stress_value'] = tau_closed
                    stress_data['max_stress_location'] = 'closed_section'
        
        # Open section stress (approximate)
        open_perimeter = self._calculate_open_perimeter()
        if open_perimeter > 0:
            # For open sections, create approximate stress distribution
            # This is a simplified visualization - exact solution requires more complex analysis
            
            # Estimate open section contribution to total torque
            J_open = (1/3) * open_perimeter * (t ** 3)
            J_total = self._get_total_J(T)
            
            if J_total > 0:
                open_torque_fraction = J_open / J_total
                T_open = T * open_torque_fraction
                
                # For rectangular strips: τ_max = 3T/(b*t²) where b is width
                # Approximate by dividing open perimeter into segments
                n_segments = min(20, len(self.outer_points))
                segment_length = open_perimeter / n_segments
                
                # Create stress points along outer boundary (simplified)
                for i in range(0, len(self.outer_points), max(1, len(self.outer_points)//n_segments)):
                    point = self.outer_points[i]
                    
                    # Approximate stress for this segment
                    # τ = 3*T_segment/(segment_length*t²)
                    T_segment = T_open / n_segments
                    tau_open = 3 * T_segment / (segment_length * t * t) if segment_length > 0 else 0
                    
                    stress_data['open_stress'].append({
                        'x': point[0],
                        'y': point[1],
                        'tau': tau_open,
                        'type': 'open',
                        'segment': i
                    })
                    
                    # Track maximum stress
                    if tau_open > stress_data['max_stress_value']:
                        stress_data['max_stress_value'] = tau_open
                        stress_data['max_stress_location'] = f'open_segment_{i}'
        
        return stress_data
    
    def _get_total_J(self, T: float) -> float:
        """Helper to get total J for stress calculations"""
        closed_J = 0
        open_J = 0
        
        # Closed contribution
        if self.inner_points:
            enclosed_area = abs(self._polygon_area_signed(self.inner_points))
            t = self.wall_thickness or self._estimate_wall_thickness()
            if enclosed_area > 0 and t > 0:
                closed_perimeter = self._polygon_perimeter(self.inner_points)
                if closed_perimeter > 0:
                    closed_J = 4 * (enclosed_area ** 2) / (closed_perimeter / t)
        
        # Open contribution  
        open_perimeter = self._calculate_open_perimeter()
        t = self.wall_thickness or self._estimate_wall_thickness()
        if open_perimeter > 0 and t > 0:
            open_J = (1/3) * open_perimeter * (t ** 3)
        
        return closed_J + open_J
    
    def create_stress_plot(self, results: Dict[str, Any], T: float, xlim=None, ylim=None) -> 'plt.Figure':
        """
        Create stress distribution visualization matching FEA plot style
        
        Args:
            results: Analysis results
            T: Applied torque 
            xlim: X-axis limits (min, max) in mm
            ylim: Y-axis limits (min, max) in mm
            
        Returns matplotlib figure showing shear stress distribution in dark theme
        """
        
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.colors import Normalize
        import matplotlib.cm as cm
        
        # Set dark theme to match FEA plots
        plt.style.use('dark_background')
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        fig.patch.set_facecolor('#0e1117')  # Streamlit dark background
        ax.set_facecolor('#0e1117')
        
        # Get stress data
        stress_data = results.get('stress_analysis', {})
        max_stress_pa = stress_data.get('max_stress_value', 1)
        max_stress_mpa = max_stress_pa / 1e6  # Convert to MPa
        
        # Set title with max stress
        ax.set_title(f'Shear Stress Distribution\nMax Stress: {max_stress_mpa:.2f} MPa', 
                    fontsize=14, fontweight='bold', color='white', pad=20)
        
        # Draw outer boundary
        if len(self.outer_points) > 2:
            outer_x = [p[0]*1000 for p in self.outer_points] + [self.outer_points[0][0]*1000]
            outer_y = [p[1]*1000 for p in self.outer_points] + [self.outer_points[0][1]*1000]
            ax.plot(outer_x, outer_y, 'white', linewidth=2, alpha=0.8)
        
        # Draw inner boundary
        if self.inner_points and len(self.inner_points) > 2:
            inner_x = [p[0]*1000 for p in self.inner_points] + [self.inner_points[0][0]*1000]
            inner_y = [p[1]*1000 for p in self.inner_points] + [self.inner_points[0][1]*1000]
            ax.plot(inner_x, inner_y, 'white', linewidth=2, alpha=0.8)
        
        # Color-code stress levels
        if max_stress_pa > 0:
            norm = Normalize(vmin=0, vmax=max_stress_mpa)  # Use MPa for normalization
            cmap = cm.get_cmap('plasma')  # Use plasma colormap like FEA plots
            
            # Plot closed section stress (constant around loop)
            for stress_point in stress_data.get('closed_stress', []):
                x, y = stress_point['x']*1000, stress_point['y']*1000
                tau_mpa = stress_point['tau'] / 1e6  # Convert to MPa
                color = cmap(norm(tau_mpa))
                ax.scatter(x, y, c=[color], s=120, marker='o', 
                          edgecolors='white', linewidth=1, alpha=0.8)
            
            # Plot open section stress (varies along length)
            for stress_point in stress_data.get('open_stress', []):
                x, y = stress_point['x']*1000, stress_point['y']*1000
                tau_mpa = stress_point['tau'] / 1e6  # Convert to MPa
                color = cmap(norm(tau_mpa))
                ax.scatter(x, y, c=[color], s=100, marker='s',
                          edgecolors='white', linewidth=1, alpha=0.8)
            
            # Add colorbar with MPa units
            sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
            cbar.set_label('Shear Stress (MPa)', rotation=270, labelpad=20, color='white')
            cbar.ax.tick_params(colors='white')
        
        # Styling to match FEA plots
        ax.set_xlabel('X (mm)', color='white', fontsize=12)
        ax.set_ylabel('Y (mm)', color='white', fontsize=12)
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.2, color='white')
        ax.set_aspect('equal')
        
        # Set axis limits - use provided limits or calculate from geometry
        if xlim and ylim:
            # Use provided zoom/pan limits
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        elif len(self.outer_points) > 2:
            # Calculate from geometry bounds with padding
            all_x = [p[0]*1000 for p in self.outer_points]
            all_y = [p[1]*1000 for p in self.outer_points]
            if self.inner_points:
                all_x.extend([p[0]*1000 for p in self.inner_points])
                all_y.extend([p[1]*1000 for p in self.inner_points])
            
            x_range = max(all_x) - min(all_x)
            y_range = max(all_y) - min(all_y)
            padding = max(x_range, y_range) * 0.1
            
            ax.set_xlim(min(all_x) - padding, max(all_x) + padding)
            ax.set_ylim(min(all_y) - padding, max(all_y) + padding)
        
        plt.tight_layout()
        return fig
    
    def _classify_section_type(self) -> str:
        """Classify the type of section"""
        
        if not self.inner_points:
            return "fully_open"
        
        outer_area = abs(self._polygon_area_signed(self.outer_points))
        inner_area = abs(self._polygon_area_signed(self.inner_points))
        
        area_ratio = inner_area / outer_area if outer_area > 0 else 0
        
        if area_ratio > 0.8:
            return "mostly_closed"
        elif area_ratio > 0.3:
            return "mixed_open_closed"
        else:
            return "mostly_open"
    
    def _estimate_wall_thickness(self) -> float:
        """Estimate wall thickness from geometry"""
        if not self.inner_points:
            return 0.0
        
        # Use existing method from MultiMethodAnalysis
        multi_analyzer = MultiMethodAnalysis(self.outer_points, self.inner_points)
        return multi_analyzer._estimate_wall_thickness()
    
    def _polygon_area_signed(self, points: List[Tuple[float, float]]) -> float:
        """Calculate signed area using shoelace formula"""
        if len(points) < 3:
            return 0.0
        
        area = 0.0
        n = len(points)
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        return area / 2.0
    
    def _polygon_perimeter(self, points: List[Tuple[float, float]]) -> float:
        """Calculate perimeter of polygon"""
        if len(points) < 2:
            return 0.0
        
        perimeter = 0.0
        n = len(points)
        for i in range(n):
            j = (i + 1) % n
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            perimeter += np.sqrt(dx*dx + dy*dy)
        return perimeter
    
    def format_results(self, results: Dict[str, Any]) -> str:
        """Format results as readable summary"""
        
        summary = []
        summary.append("="*60)
        summary.append("MIXED SECTION TORSIONAL ANALYSIS")
        summary.append("="*60)
        
        # Geometry
        geom = results['geometry']
        summary.append(f"\n📐 SECTION GEOMETRY:")
        summary.append(f"  Type: {geom['section_type'].replace('_', ' ').title()}")
        summary.append(f"  Net Area: {geom['net_area']*1e6:.1f} mm²")
        summary.append(f"  Closed Cavity Area: {geom['closed_cavity_area']*1e6:.1f} mm²")
        summary.append(f"  Open Perimeter: {geom['open_perimeter']*1000:.1f} mm")
        summary.append(f"  Wall Thickness: {geom['wall_thickness_estimated']*1000:.1f} mm")
        
        # Component analysis
        summary.append(f"\n🔒 CLOSED PART ANALYSIS:")
        closed = results['closed_contribution']
        if closed.get('J', 0) and closed['J'] > 0:
            summary.append(f"  Method: {closed['method']}")
            summary.append(f"  J = {closed['J']:.6e} m⁴")
            summary.append(f"  {closed.get('note', '')}")
        else:
            summary.append(f"  {closed.get('note', 'No closed contribution')}")
        
        summary.append(f"\n🔓 OPEN PARTS ANALYSIS:")
        open_part = results['open_contribution']
        if open_part.get('J', 0) and open_part['J'] > 0:
            summary.append(f"  Method: {open_part['method']}")
            summary.append(f"  J = {open_part['J']:.6e} m⁴")
            summary.append(f"  {open_part.get('note', '')}")
        else:
            summary.append(f"  {open_part.get('note', 'No open contribution')}")
        
        # Total results
        if 'total' in results and results['total']:
            total = results['total']
            summary.append(f"\n🎯 TOTAL RESULTS:")
            summary.append(f"  Total J = {total['J']:.6e} m⁴")
            summary.append(f"  Closed fraction: {total['closed_fraction']*100:.1f}%")
            summary.append(f"  Open fraction: {total['open_fraction']*100:.1f}%")
            summary.append(f"  Twist angle: {total['theta_deg']:.6f}°")
            
            # Engineering interpretation
            summary.append(f"\n💡 ENGINEERING INSIGHT:")
            if total['closed_fraction'] > 0.9:
                summary.append(f"  Dominated by closed cavity - Bredt theory applies")
            elif total['open_fraction'] > 0.9:
                summary.append(f"  Dominated by open parts - open thin-wall theory applies")
            else:
                summary.append(f"  Mixed behavior - both contributions significant")
        
        return "\n".join(summary)

def test_mixed_section():
    """Test the mixed section analysis"""
    
    # Example: Your ship section (simplified)
    # Outer boundary
    outer_points_mm = [
        (0,0), (199.41,25.64), (368.06,92.10), (491.03,194.82), (614.27,264.61),
        (617.16,267.06), (702.58,849.67), (705.06,852.2), (716.01,852.24),
        (716.01,866.64), (706.56,866.64), (703.56,869.45), (682.47,1343.82),
        (674.50,1353.16), (620.55,1353.37), (614.56,1347.68), (610.93,1278.37),
        (613.92,1278.21), (617.56,1347.52), (620.55,1350.37), (672.48,1350.37),
        (679.47,1343.80), (700.57,869.29), (699.62,850.11), (623.06,327.60),
        (-623.06,327.60), (-699.62,850.11), (-700.57,869.29), (-679.47,1343.80),
        (-672.48,1350.37), (-620.55,1350.37), (-617.56,1347.52), (-613.92,1278.21),
        (-610.93,1278.37), (-614.56,1347.68), (-620.55,1353.37), (-674.50,1353.16),
        (-682.47,1343.82), (-703.56,869.45), (-706.56,866.64), (-716.01,866.64),
        (-716.01,852.24), (-705.06,852.2), (-702.58,849.67), (-617.16,267.06),
        (-614.27,264.61), (-491.03,194.82), (-368.06,92.10), (-199.41,25.64)
    ]
    
    # Inner boundary (closed cavity)
    inner_points_mm = [
        (0.00,3.01), (-198.70,28.55), (-366.12,94.41), (-488.47,196.55),
        (-614.21,267.61), (-622.62,324.60), (622.62,324.60), (614.21,267.61),
        (488.47,196.55), (366.12,94.41), (198.70,28.55)
    ]
    
    # Convert to meters
    outer_points_m = [(p[0]/1000, p[1]/1000) for p in outer_points_mm]
    inner_points_m = [(p[0]/1000, p[1]/1000) for p in inner_points_mm]
    
    # Initialize mixed section analysis
    mixed_analyzer = MixedSectionAnalysis(
        outer_points_m, 
        inner_points_m, 
        wall_thickness=0.003  # 3mm wall thickness
    )
    
    # Analysis parameters
    G = 80000e6  # Pa
    T = 1000     # Nm  
    L = 2.0      # m
    
    # Run analysis
    results = mixed_analyzer.analyze_mixed_section(G, T, L)
    
    # Display results
    print(mixed_analyzer.format_results(results))

if __name__ == "__main__":
    test_mixed_section()
