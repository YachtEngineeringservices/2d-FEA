#!/usr/bin/env python3
"""
Multi-Method Torsional Analysis
Compares different engineering approaches for calculating torsional properties
"""

import numpy as np
import logging
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)

class MultiMethodAnalysis:
    """
    Comprehensive torsional analysis using multiple methods:
    1. FEA (Saint-Venant) - current implementation
    2. Thin-walled (Bredt's formula)
    3. Thick-walled approximation
    4. Geometric analysis
    """
    
    def __init__(self, outer_points: List[Tuple[float, float]], 
                 inner_points: List[Tuple[float, float]] = None):
        """
        Initialize with geometry points in meters
        """
        self.outer_points = outer_points
        self.inner_points = inner_points or []
        self.results = {}
        
    def analyze_all_methods(self, G: float, T: float, L: float, 
                          fea_J: float = None) -> Dict[str, Any]:
        """
        Run all analysis methods and compare results
        
        Args:
            G: Shear modulus (Pa)
            T: Applied torque (Nm)
            L: Length (m)
            fea_J: FEA result for comparison (m⁴)
        """
        results = {
            'geometry': self._analyze_geometry(),
            'thin_walled': self._analyze_thin_walled(),
            'thick_walled': self._analyze_thick_walled(),
            'comparison': {}
        }
        
        # Add FEA results if provided
        if fea_J is not None:
            results['fea'] = {
                'J': fea_J,
                'k': G * fea_J / L,
                'theta_rad': T / (G * fea_J / L),
                'theta_deg': np.rad2deg(T / (G * fea_J / L)),
                'method': 'Saint-Venant FEA'
            }
        
        # Calculate performance metrics for all methods
        for method_name, method_data in results.items():
            if method_name in ['geometry', 'comparison']:
                continue
                
            if 'J' in method_data and method_data['J'] is not None and method_data['J'] > 0:
                J = method_data['J']
                k = G * J / L  # Torsional stiffness
                theta_rad = T / k
                theta_deg = np.rad2deg(theta_rad)
                
                method_data.update({
                    'k': k,
                    'theta_rad': theta_rad,
                    'theta_deg': theta_deg,
                    'max_shear_stress': self._estimate_max_shear_stress(J, T)
                })
        
        # Create comparison matrix
        results['comparison'] = self._create_comparison(results)
        
        self.results = results
        return results
    
    def _analyze_geometry(self) -> Dict[str, Any]:
        """Analyze basic geometric properties"""
        
        # Calculate areas using shoelace formula
        outer_area = abs(self._polygon_area_signed(self.outer_points))
        inner_area = abs(self._polygon_area_signed(self.inner_points)) if self.inner_points else 0
        net_area = outer_area - inner_area
        
        # Calculate perimeters
        outer_perimeter = self._polygon_perimeter(self.outer_points)
        inner_perimeter = self._polygon_perimeter(self.inner_points) if self.inner_points else 0
        total_perimeter = outer_perimeter + inner_perimeter
        
        # Calculate approximate wall thickness
        wall_thickness = self._estimate_wall_thickness()
        
        # Bounding box
        all_points = self.outer_points + (self.inner_points or [])
        x_coords = [p[0] for p in all_points]
        y_coords = [p[1] for p in all_points]
        
        return {
            'area': net_area,  # Add the main area key that other functions expect
            'outer_area': outer_area,
            'inner_area': inner_area,
            'net_area': net_area,
            'outer_perimeter': outer_perimeter,
            'inner_perimeter': inner_perimeter,
            'total_perimeter': total_perimeter,
            'wall_thickness': wall_thickness,
            'bounding_box': {
                'width': max(x_coords) - min(x_coords),
                'height': max(y_coords) - min(y_coords),
                'x_range': (min(x_coords), max(x_coords)),
                'y_range': (min(y_coords), max(y_coords))
            }
        }
    
    def _analyze_thin_walled(self) -> Dict[str, Any]:
        """
        Thin-walled analysis using Bredt's formula
        J = 4A²/∮(ds/t) where A is enclosed area, ds/t is perimeter/thickness integral
        """
        
        if not self.inner_points:
            # Solid section - not applicable
            return {
                'J': None,
                'method': 'Bredt (not applicable - solid section)',
                'note': 'Thin-walled theory requires hollow section'
            }
        
        # Use average enclosed area (between outer and inner boundaries)
        outer_area = abs(self._polygon_area_signed(self.outer_points))
        inner_area = abs(self._polygon_area_signed(self.inner_points))
        enclosed_area = (outer_area + inner_area) / 2  # Average area
        
        # Estimate wall thickness
        wall_thickness = self._estimate_wall_thickness()
        
        if wall_thickness <= 0:
            return {
                'J': None,
                'method': 'Bredt (failed - cannot determine wall thickness)',
                'note': 'Unable to estimate wall thickness'
            }
        
        # Calculate perimeter integral ∮(ds/t)
        # Assuming uniform thickness
        outer_perimeter = self._polygon_perimeter(self.outer_points)
        inner_perimeter = self._polygon_perimeter(self.inner_points)
        total_perimeter = outer_perimeter + inner_perimeter
        
        perimeter_integral = total_perimeter / wall_thickness
        
        # Bredt's formula: J = 4A²/∮(ds/t)
        J_bredt = 4 * (enclosed_area ** 2) / perimeter_integral
        
        return {
            'J': J_bredt,
            'method': 'Bredt thin-walled formula',
            'enclosed_area': enclosed_area,
            'wall_thickness': wall_thickness,
            'perimeter_integral': perimeter_integral,
            'outer_perimeter': outer_perimeter,
            'inner_perimeter': inner_perimeter
        }
    
    def _analyze_thick_walled(self) -> Dict[str, Any]:
        """
        Thick-walled approximation using polar moment of inertia
        J ≈ Ix + Iy for thin sections, or more sophisticated for thick sections
        """
        
        # Calculate second moments of area
        Ix = self._calculate_second_moment_x()
        Iy = self._calculate_second_moment_y()
        
        # For thin-walled sections: J ≈ Ix + Iy
        J_polar_approx = Ix + Iy
        
        # For thick sections, use more accurate approximation
        # This is a simplified approach - real thick-walled needs more complex analysis
        net_area = self._analyze_geometry()['net_area']
        
        # Approximate correction factor based on geometry
        if self.inner_points:
            # Hollow section
            outer_area = abs(self._polygon_area_signed(self.outer_points))
            inner_area = abs(self._polygon_area_signed(self.inner_points))
            area_ratio = inner_area / outer_area
            # Correction factor decreases with increasing hole size
            correction_factor = 0.8 + 0.2 * (1 - area_ratio)
        else:
            # Solid section
            correction_factor = 0.9  # Typical for solid irregular sections
        
        J_thick_walled = J_polar_approx * correction_factor
        
        return {
            'J': J_thick_walled,
            'method': 'Thick-walled approximation (Ix + Iy)',
            'Ix': Ix,
            'Iy': Iy,
            'J_polar_approx': J_polar_approx,
            'correction_factor': correction_factor,
            'net_area': net_area
        }
    
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
    
    def _estimate_wall_thickness(self) -> float:
        """
        Estimate wall thickness by comparing outer and inner boundaries
        """
        if not self.inner_points:
            return 0.0  # Solid section
        
        # Method 1: Average distance between corresponding points
        if len(self.outer_points) == len(self.inner_points):
            distances = []
            for i in range(len(self.outer_points)):
                dx = self.outer_points[i][0] - self.inner_points[i][0]
                dy = self.outer_points[i][1] - self.inner_points[i][1]
                distances.append(np.sqrt(dx*dx + dy*dy))
            return np.mean(distances)
        
        # Method 2: Difference in equivalent radii
        outer_area = abs(self._polygon_area_signed(self.outer_points))
        inner_area = abs(self._polygon_area_signed(self.inner_points))
        
        outer_radius = np.sqrt(outer_area / np.pi)
        inner_radius = np.sqrt(inner_area / np.pi)
        
        return outer_radius - inner_radius
    
    def _calculate_second_moment_x(self) -> float:
        """Calculate second moment of area about x-axis"""
        # Simplified calculation using triangulation
        # This is approximate - exact calculation requires more complex integration
        
        outer_Ix = self._polygon_second_moment_x(self.outer_points)
        inner_Ix = self._polygon_second_moment_x(self.inner_points) if self.inner_points else 0
        
        return outer_Ix - inner_Ix
    
    def _calculate_second_moment_y(self) -> float:
        """Calculate second moment of area about y-axis"""
        outer_Iy = self._polygon_second_moment_y(self.outer_points)
        inner_Iy = self._polygon_second_moment_y(self.inner_points) if self.inner_points else 0
        
        return outer_Iy - inner_Iy
    
    def _polygon_second_moment_x(self, points: List[Tuple[float, float]]) -> float:
        """Approximate second moment of area about x-axis for polygon"""
        if len(points) < 3:
            return 0.0
        
        # Using triangulation from centroid
        cx, cy = self._polygon_centroid(points)
        
        Ix = 0.0
        n = len(points)
        for i in range(n):
            j = (i + 1) % n
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            # Triangle area (with centroid)
            area_triangle = 0.5 * abs((x1-cx)*(y2-cy) - (x2-cx)*(y1-cy))
            
            # Triangle centroid
            tri_cy = (cy + y1 + y2) / 3
            
            # Contribution to Ix
            Ix += area_triangle * (tri_cy ** 2)
        
        return Ix
    
    def _polygon_second_moment_y(self, points: List[Tuple[float, float]]) -> float:
        """Approximate second moment of area about y-axis for polygon"""
        if len(points) < 3:
            return 0.0
        
        # Using triangulation from centroid
        cx, cy = self._polygon_centroid(points)
        
        Iy = 0.0
        n = len(points)
        for i in range(n):
            j = (i + 1) % n
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            # Triangle area (with centroid)
            area_triangle = 0.5 * abs((x1-cx)*(y2-cy) - (x2-cx)*(y1-cy))
            
            # Triangle centroid
            tri_cx = (cx + x1 + x2) / 3
            
            # Contribution to Iy
            Iy += area_triangle * (tri_cx ** 2)
        
        return Iy
    
    def _polygon_centroid(self, points: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Calculate centroid of polygon"""
        if len(points) < 3:
            return (0.0, 0.0)
        
        area = self._polygon_area_signed(points)
        if abs(area) < 1e-12:
            return (0.0, 0.0)
        
        cx = cy = 0.0
        n = len(points)
        for i in range(n):
            j = (i + 1) % n
            factor = points[i][0] * points[j][1] - points[j][0] * points[i][1]
            cx += (points[i][0] + points[j][0]) * factor
            cy += (points[i][1] + points[j][1]) * factor
        
        factor = 1.0 / (6.0 * area)
        return (cx * factor, cy * factor)
    
    def _estimate_max_shear_stress(self, J: float, T: float) -> float:
        """
        Estimate maximum shear stress
        For irregular sections, this is approximate
        """
        if J <= 0 or J is None:
            return 0.0
        
        # Estimate maximum radius (distance from centroid to boundary)
        geometry = self._analyze_geometry()
        max_radius = max(
            geometry['bounding_box']['width'],
            geometry['bounding_box']['height']
        ) / 2
        
        # τ = T*r/J
        return T * max_radius / J
    
    def _create_comparison(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create comparison matrix between methods"""
        
        methods = []
        J_values = []
        theta_values = []
        stress_values = []
        
        for method_name, method_data in results.items():
            if method_name in ['geometry', 'comparison']:
                continue
            
            if 'J' in method_data and method_data['J'] is not None:
                methods.append(method_name)
                J_values.append(method_data['J'])
                theta_values.append(method_data.get('theta_deg', 0))
                stress_values.append(method_data.get('max_shear_stress', 0))
        
        if len(J_values) < 2:
            return {'note': 'Insufficient methods for comparison'}
        
        # Find reference (typically FEA if available, otherwise first method)
        ref_idx = 0
        if 'fea' in [m for m in methods]:
            ref_idx = methods.index('fea')
        
        ref_J = J_values[ref_idx]
        ref_method = methods[ref_idx]
        
        comparison = {
            'reference_method': ref_method,
            'reference_J': ref_J,
            'method_comparison': []
        }
        
        for i, method in enumerate(methods):
            if J_values[i] > 0:
                ratio = J_values[i] / ref_J
                agreement = 'excellent' if 0.5 <= ratio <= 2.0 else \
                           'good' if 0.1 <= ratio <= 10.0 else \
                           'poor'
            else:
                ratio = 0
                agreement = 'failed'
            
            comparison['method_comparison'].append({
                'method': method,
                'J': J_values[i],
                'ratio_to_reference': ratio,
                'agreement': agreement,
                'theta_deg': theta_values[i] if i < len(theta_values) else 0,
                'max_stress_pa': stress_values[i] if i < len(stress_values) else 0
            })
        
        return comparison

    def format_results_summary(self) -> str:
        """Format results as readable summary"""
        if not self.results:
            return "No analysis results available"
        
        summary = []
        summary.append("="*60)
        summary.append("MULTI-METHOD TORSIONAL ANALYSIS SUMMARY")
        summary.append("="*60)
        
        # Geometry summary
        geom = self.results['geometry']
        summary.append(f"\n📐 GEOMETRY:")
        summary.append(f"  Net Area: {geom['net_area']*1e6:.1f} mm² ({geom['net_area']:.6f} m²)")
        summary.append(f"  Outer Area: {geom['outer_area']*1e6:.1f} mm²")
        if geom['inner_area'] > 0:
            summary.append(f"  Inner Area: {geom['inner_area']*1e6:.1f} mm²")
            summary.append(f"  Wall Thickness: {geom['wall_thickness']*1000:.1f} mm")
        summary.append(f"  Bounding Box: {geom['bounding_box']['width']*1000:.1f} × {geom['bounding_box']['height']*1000:.1f} mm")
        
        # Method results
        summary.append(f"\n🔬 METHOD RESULTS:")
        
        for method_name, method_data in self.results.items():
            if method_name in ['geometry', 'comparison']:
                continue
                
            if 'J' in method_data and method_data['J'] is not None:
                summary.append(f"\n  {method_data.get('method', method_name.upper())}:")
                summary.append(f"    J = {method_data['J']:.6e} m⁴")
                summary.append(f"    θ = {method_data.get('theta_deg', 0):.3f}°")
                summary.append(f"    τ_max = {method_data.get('max_shear_stress', 0)/1e6:.1f} MPa")
            else:
                summary.append(f"\n  {method_data.get('method', method_name.upper())}: {method_data.get('note', 'Failed')}")
        
        # Comparison
        if 'comparison' in self.results and 'method_comparison' in self.results['comparison']:
            comp = self.results['comparison']
            summary.append(f"\n⚖️  METHOD COMPARISON:")
            summary.append(f"  Reference: {comp['reference_method']} (J = {comp['reference_J']:.6e} m⁴)")
            
            for method_comp in comp['method_comparison']:
                if method_comp['method'] != comp['reference_method']:
                    ratio = method_comp['ratio_to_reference']
                    agreement = method_comp['agreement']
                    summary.append(f"  {method_comp['method']}: {ratio:.2f}× reference ({agreement})")
        
        return "\n".join(summary)
