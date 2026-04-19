"""
Cross-Section Post-Processor
=============================

Flux-volume weighted cross-section collapse and homogenization.

Produces:
- 1-group cross sections for each material region
- 1-group cross sections for the whole problem (full homogenization)
- 2-group cross sections (groups 1-4 → G1, groups 5-8 → G2) with scattering matrix

Cross-section types:
- Total (Σ_t)
- Absorption (Σ_a)
- Scattering (Σ_s or scattering matrix)
- Transport (Σ_tr = Σ_t - μ̄·Σ_s)
- Fission (Σ_f)
- Nu-fission (νΣ_f)
- Kappa-fission (κΣ_f)
- Chi (χ) - fission spectrum

Weighting formula:
    Σ_collapsed = Σ_g Σ_i (Σ_{g,i} · φ_{g,i} · V_i) / Σ_g Σ_i (φ_{g,i} · V_i)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from mpact_geometry import OneDimensionalCartesianGeometryAndMesh
from sn_transport import LDFESNTransportSolver


@dataclass
class CollapsedXS:
    """Container for collapsed cross-section data."""
    name: str
    n_groups: int
    
    # Cross sections
    sigma_t: np.ndarray = None       # Total
    sigma_a: np.ndarray = None       # Absorption
    sigma_s: np.ndarray = None       # Total scattering (diagonal sum for multi-group)
    sigma_tr: np.ndarray = None      # Transport
    sigma_f: np.ndarray = None       # Fission
    nu_sigma_f: np.ndarray = None    # Nu-fission
    kappa_sigma_f: np.ndarray = None # Kappa-fission (energy release)
    
    # Scattering matrix (n_groups x n_groups)
    scatter_matrix: np.ndarray = None
    
    # Fission spectrum
    chi: np.ndarray = None
    
    # Flux used for weighting
    flux: np.ndarray = None
    
    # Volume
    volume: float = 0.0
    
    def __post_init__(self):
        if self.n_groups > 0:
            self.sigma_t = np.zeros(self.n_groups)
            self.sigma_a = np.zeros(self.n_groups)
            self.sigma_s = np.zeros(self.n_groups)
            self.sigma_tr = np.zeros(self.n_groups)
            self.sigma_f = np.zeros(self.n_groups)
            self.nu_sigma_f = np.zeros(self.n_groups)
            self.kappa_sigma_f = np.zeros(self.n_groups)
            self.scatter_matrix = np.zeros((self.n_groups, self.n_groups))
            self.chi = np.zeros(self.n_groups)
            self.flux = np.zeros(self.n_groups)
    
    def print_summary(self, indent: str = ""):
        """Print a formatted summary of the collapsed cross sections."""
        print(f"{indent}{'='*60}")
        print(f"{indent}{self.name}")
        print(f"{indent}{'='*60}")
        print(f"{indent}  Groups: {self.n_groups}")
        print(f"{indent}  Volume: {self.volume:.6e} cm")
        
        print(f"\n{indent}  {'Group':<8} {'Σ_t':<12} {'Σ_a':<12} {'Σ_s':<12} {'Σ_tr':<12}")
        print(f"{indent}  {'-'*52}")
        for g in range(self.n_groups):
            print(f"{indent}  {g+1:<8} {self.sigma_t[g]:<12.6e} {self.sigma_a[g]:<12.6e} "
                  f"{self.sigma_s[g]:<12.6e} {self.sigma_tr[g]:<12.6e}")
        
        print(f"\n{indent}  {'Group':<8} {'Σ_f':<12} {'νΣ_f':<12} {'κΣ_f':<12} {'χ':<12}")
        print(f"{indent}  {'-'*52}")
        for g in range(self.n_groups):
            print(f"{indent}  {g+1:<8} {self.sigma_f[g]:<12.6e} {self.nu_sigma_f[g]:<12.6e} "
                  f"{self.kappa_sigma_f[g]:<12.6e} {self.chi[g]:<12.6e}")
        
        if self.n_groups > 1:
            print(f"\n{indent}  Scattering Matrix Σ_s(g→g'):")
            print(f"{indent}  " + " "*8 + "".join([f"  To G{g+1:<5}" for g in range(self.n_groups)]))
            for g_from in range(self.n_groups):
                row = f"{indent}  From G{g_from+1:<3}"
                for g_to in range(self.n_groups):
                    row += f"{self.scatter_matrix[g_from, g_to]:<12.6e}"
                print(row)
        
        print(f"\n{indent}  Flux (for weighting):")
        for g in range(self.n_groups):
            print(f"{indent}    Group {g+1}: φ = {self.flux[g]:.6e}")


class CrossSectionPostProcessor:
    """
    Post-processor for flux-volume weighted cross-section collapse.
    
    Takes a solved transport problem and produces collapsed cross sections.
    """
    
    def __init__(self, solver: LDFESNTransportSolver):
        """
        Initialize post-processor with a solved transport problem.
        
        Args:
            solver: Solved LDFESNTransportSolver instance
        """
        self.solver = solver
        self.geom = solver.geom
        self.n_cells = solver.n_cells
        self.n_groups = solver.n_groups
        
        # Get flux and geometry data
        self.scalar_flux = solver.solution.scalar_flux  # (n_cells, n_groups)
        self.cell_volumes = np.array([cell.width for cell in self.geom.cells])  # 1D: volume = width
        self.cell_densities = [cell.material.density for cell in self.geom.cells]

        # Extract fine-group cross sections per cell
        self._extract_fine_group_xs()
        
        # Standard fission spectrum (8-group MPACT)
        self.chi_fine = np.array([0.584349, 0.415378, 0.000272, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Average cosine for transport XS (P1 scattering approx)
        # For hydrogen-dominated systems, μ̄ ≈ 2/(3A) where A is mass number
        # We'll compute this from the scattering data if available, or use 0 (isotropic)
        self.mu_bar = np.zeros((self.n_cells, self.n_groups))
        
        print(f"\nCross-Section Post-Processor initialized")
        print(f"  Cells: {self.n_cells}, Groups: {self.n_groups}")
        print(f"  Total volume: {self.cell_volumes.sum():.6f} cm")
    
    def _extract_fine_group_xs(self):
        """Extract all fine-group cross sections from the geometry."""
        
        # Initialize arrays (n_cells, n_groups)
        self.sigma_t = np.zeros((self.n_cells, self.n_groups))
        self.sigma_a = np.zeros((self.n_cells, self.n_groups))
        self.sigma_f = np.zeros((self.n_cells, self.n_groups))
        self.nu_sigma_f = np.zeros((self.n_cells, self.n_groups))
        self.kappa_sigma_f = np.zeros((self.n_cells, self.n_groups))
        
        # Scattering matrix (n_cells, n_groups, n_groups)
        self.scatter_matrix_fine = np.zeros((self.n_cells, self.n_groups, self.n_groups))
        
        for i, cell in enumerate(self.geom.cells):
            mat = cell.material
            
            # Total and absorption
            self.sigma_t[i, :] = mat.get_macroscopic_xs('total')
            self.sigma_a[i, :] = mat.get_macroscopic_xs('absorption')
            
            # Fission cross sections
            try:
                self.sigma_f[i, :] = mat.get_macroscopic_xs('fission')
            except:
                pass
            
            # Nu-fission: try to get from library, otherwise estimate
            try:
                nu_f = mat.get_macroscopic_xs('nu_fission')
                if np.any(nu_f > 0):
                    self.nu_sigma_f[i, :] = nu_f
                else:
                    # Estimate: ν ≈ 2.5 for thermal, 2.8 for fast
                    nu_approx = np.array([2.8, 2.7, 2.6, 2.5, 2.5, 2.5, 2.5, 2.5])
                    self.nu_sigma_f[i, :] = nu_approx * self.sigma_f[i, :]
            except:
                nu_approx = np.array([2.8, 2.7, 2.6, 2.5, 2.5, 2.5, 2.5, 2.5])
                self.nu_sigma_f[i, :] = nu_approx * self.sigma_f[i, :]
            
            # Kappa-fission: try to get from library, otherwise estimate
            try:
                kappa_f = mat.get_macroscopic_xs('kappa_fission')
                if np.any(kappa_f > 0):
                    self.kappa_sigma_f[i, :] = kappa_f
                else:
                    # Estimate: κ ≈ 200 MeV per fission (3.2e-11 J)
                    self.kappa_sigma_f[i, :] = 200.0 * self.sigma_f[i, :]
            except:
                self.kappa_sigma_f[i, :] = 200.0 * self.sigma_f[i, :]
            
            # Scattering matrix
            for comp in mat.components.values():
                if hasattr(comp.nuclide, 'scatter_matrix'):
                    self.scatter_matrix_fine[i, :, :] += (
                        comp.number_density * comp.nuclide.scatter_matrix
                    )
        
        # Total scattering (sum over outgoing groups)
        self.sigma_s = self.scatter_matrix_fine.sum(axis=2)  # Sum over g_to
    
    def collapse_to_1group(self, cell_indices: Optional[List[int]] = None,
                           name: str = "1-Group") -> CollapsedXS:
        """
        Collapse to 1 energy group using flux-volume weighting.
        
        Args:
            cell_indices: List of cell indices to include (None = all cells)
            name: Name for the collapsed data
        
        Returns:
            CollapsedXS object with 1-group data
        """
        if cell_indices is None:
            cell_indices = list(range(self.n_cells))
        
        xs = CollapsedXS(name=name, n_groups=1)
        
        # Compute flux-volume weights
        # φ_g V_i for selected cells
        phi_V = np.zeros(self.n_groups)
        for i in cell_indices:
            phi_V += self.scalar_flux[i, :] * self.cell_volumes[i]
        
        total_phi_V = phi_V.sum()
        xs.volume = sum(self.cell_volumes[i] for i in cell_indices)
        # density = sum(self.cell_densities[i] for i in cell_indices) 
        mass_number = sum(self.cell_volumes[i]*self.cell_densities[i] for i in cell_indices)
        # print(f'volumes\n{self.cell_volumes}')
        # print(f'densities\n{self.cell_densities}')

        # print(f'mass numba {mass_number}')
        # mass_number = xs.volume*density 
        if total_phi_V < 1e-30:
            print(f"  Warning: Near-zero flux in {name}, using uniform weights")
            phi_V = np.ones(self.n_groups) * xs.volume / self.n_groups
            total_phi_V = phi_V.sum()
        
        # Collapse each cross section type
        # Σ_1g = Σ_g (Σ_i Σ_g,i φ_i,g V_i) / Σ_g (Σ_i φ_i,g V_i)
        
        sigma_t_weighted = np.zeros(self.n_groups)
        sigma_a_weighted = np.zeros(self.n_groups)
        sigma_s_weighted = np.zeros(self.n_groups)
        sigma_f_weighted = np.zeros(self.n_groups)
        nu_sigma_f_weighted = np.zeros(self.n_groups)
        kappa_sigma_f_weighted = np.zeros(self.n_groups)
        
        for i in cell_indices:
            V_i = self.cell_volumes[i]
            
            for g in range(self.n_groups):
                weight = self.scalar_flux[i, g] * V_i
                sigma_t_weighted[g] += self.sigma_t[i, g] * weight
                sigma_a_weighted[g] += self.sigma_a[i, g] * weight
                sigma_s_weighted[g] += self.sigma_s[i, g] * weight
                sigma_f_weighted[g] += self.sigma_f[i, g] * weight
                nu_sigma_f_weighted[g] += self.nu_sigma_f[i, g] * weight
                kappa_sigma_f_weighted[g] += self.kappa_sigma_f[i, g] * weight
                
        # Final 1-group values
        xs.sigma_t[0] = sigma_t_weighted.sum() / total_phi_V
        xs.sigma_a[0] = sigma_a_weighted.sum() / total_phi_V
        xs.sigma_s[0] = sigma_s_weighted.sum() / total_phi_V
        xs.sigma_f[0] = sigma_f_weighted.sum() / total_phi_V
        xs.nu_sigma_f[0] = nu_sigma_f_weighted.sum() / total_phi_V
        xs.kappa_sigma_f[0] = kappa_sigma_f_weighted.sum() / total_phi_V
        
        # Transport cross section: Σ_tr = Σ_t - μ̄·Σ_s
        # For isotropic scattering (μ̄ = 0): Σ_tr = Σ_t
        # Better approximation: Σ_tr ≈ Σ_a + Σ_s(1 - μ̄) ≈ Σ_t - (2/3A)Σ_s for hydrogen
        xs.sigma_tr[0] = xs.sigma_t[0]  # Isotropic approximation

        xs.sigma_tr[0] = xs.sigma_t[0] - (2/(3*mass_number))*xs.sigma_s[0]
        # 1-group scattering matrix is just the scalar value
        xs.scatter_matrix[0, 0] = xs.sigma_s[0]
        
        # Chi: for 1-group, χ = 1.0 (all fission neutrons born in the one group)
        xs.chi[0] = 1.0
        
        # Store flux
        xs.flux[0] = total_phi_V / xs.volume
        
        return xs
    
    def collapse_to_2group(self, cell_indices: Optional[List[int]] = None,
                           name: str = "2-Group",
                           group_boundaries: Tuple[int, int] = (4, 8)) -> CollapsedXS:
        """
        Collapse to 2 energy groups using flux-volume weighting.
        
        Default: Groups 1-4 → G1 (fast), Groups 5-8 → G2 (thermal)
        
        Args:
            cell_indices: List of cell indices to include (None = all cells)
            name: Name for the collapsed data
            group_boundaries: (end of G1, end of G2) in fine-group numbering
        
        Returns:
            CollapsedXS object with 2-group data
        """
        if cell_indices is None:
            cell_indices = list(range(self.n_cells))
        
        g1_end, g2_end = group_boundaries  # e.g., (4, 8) means G1=[0:4], G2=[4:8]
        
        # Fine group ranges (0-indexed)
        g1_range = range(0, g1_end)           # Groups 1-4 → indices 0-3
        g2_range = range(g1_end, g2_end)      # Groups 5-8 → indices 4-7
        coarse_ranges = [g1_range, g2_range]
        
        xs = CollapsedXS(name=name, n_groups=2)
        xs.volume = sum(self.cell_volumes[i] for i in cell_indices)
        
        # Compute flux-volume weights for each coarse group
        phi_V_coarse = np.zeros(2)
        phi_V_fine = np.zeros(self.n_groups)
        
        for i in cell_indices:
            V_i = self.cell_volumes[i]
            for g in range(self.n_groups):
                phi_V_fine[g] += self.scalar_flux[i, g] * V_i
        
        for G, g_range in enumerate(coarse_ranges):
            phi_V_coarse[G] = sum(phi_V_fine[g] for g in g_range)
        
        # Collapse cross sections for each coarse group
        for G, g_range in enumerate(coarse_ranges):
            if phi_V_coarse[G] < 1e-30:
                print(f"  Warning: Near-zero flux in coarse group {G+1}")
                continue
            
            # Weighted sums over fine groups and cells
            sigma_t_sum = 0.0
            sigma_a_sum = 0.0
            sigma_s_sum = 0.0
            sigma_f_sum = 0.0
            nu_sigma_f_sum = 0.0
            kappa_sigma_f_sum = 0.0
            
            for i in cell_indices:
                V_i = self.cell_volumes[i]
                for g in g_range:
                    weight = self.scalar_flux[i, g] * V_i
                    sigma_t_sum += self.sigma_t[i, g] * weight
                    sigma_a_sum += self.sigma_a[i, g] * weight
                    sigma_s_sum += self.sigma_s[i, g] * weight
                    sigma_f_sum += self.sigma_f[i, g] * weight
                    nu_sigma_f_sum += self.nu_sigma_f[i, g] * weight
                    kappa_sigma_f_sum += self.kappa_sigma_f[i, g] * weight
            
            xs.sigma_t[G] = sigma_t_sum / phi_V_coarse[G]
            xs.sigma_a[G] = sigma_a_sum / phi_V_coarse[G]
            xs.sigma_s[G] = sigma_s_sum / phi_V_coarse[G]
            xs.sigma_f[G] = sigma_f_sum / phi_V_coarse[G]
            xs.nu_sigma_f[G] = nu_sigma_f_sum / phi_V_coarse[G]
            xs.kappa_sigma_f[G] = kappa_sigma_f_sum / phi_V_coarse[G]
            xs.sigma_tr[G] = xs.sigma_t[G]  # Isotropic approximation
            xs.flux[G] = phi_V_coarse[G] / xs.volume
        
        # 2x2 Scattering matrix
        # Σ_s(G→G') = Σ_{g∈G} Σ_{g'∈G'} Σ_i [Σ_s,i(g→g') φ_i,g V_i] / Σ_{g∈G} Σ_i [φ_i,g V_i]
        for G_from, g_from_range in enumerate(coarse_ranges):
            if phi_V_coarse[G_from] < 1e-30:
                continue
            
            for G_to, g_to_range in enumerate(coarse_ranges):
                scatter_sum = 0.0
                
                for i in cell_indices:
                    V_i = self.cell_volumes[i]
                    for g_from in g_from_range:
                        weight = self.scalar_flux[i, g_from] * V_i
                        for g_to in g_to_range:
                            scatter_sum += self.scatter_matrix_fine[i, g_from, g_to] * weight
                
                xs.scatter_matrix[G_from, G_to] = scatter_sum / phi_V_coarse[G_from]
        
        # Chi: collapse fission spectrum
        # χ_G = Σ_{g∈G} χ_g (simple sum since χ is already normalized)
        for G, g_range in enumerate(coarse_ranges):
            xs.chi[G] = sum(self.chi_fine[g] for g in g_range)
        
        # Renormalize chi to sum to 1
        chi_sum = xs.chi.sum()
        if chi_sum > 0:
            xs.chi /= chi_sum
        
        return xs
    
    def process_all(self) -> Dict[str, CollapsedXS]:
        """
        Process all cross-section collapses:
        - 1-group for each material region
        - 1-group for whole problem
        - 2-group for each material region
        - 2-group for whole problem
        
        Returns:
            Dictionary of CollapsedXS objects
        """
        results = {}
        
        print("\n" + "="*70)
        print("CROSS-SECTION COLLAPSE AND HOMOGENIZATION")
        print("="*70)
        
        # Get region information
        regions = self.geom.regions
        
        # 1-group by region
        print("\n--- 1-Group Cross Sections by Region ---")
        for region_idx, region in enumerate(regions):
            cell_indices = [c.index for c in self.geom.cells if c.region_index == region_idx]
            if cell_indices:
                name = f"1-Group: {region.material.name}"
                results[f'1g_region_{region_idx}'] = self.collapse_to_1group(
                    cell_indices=cell_indices, name=name
                )
                results[f'1g_region_{region_idx}'].print_summary("  ")
        
        # 1-group whole problem
        print("\n--- 1-Group Homogenized (Whole Problem) ---")
        results['1g_total'] = self.collapse_to_1group(name="1-Group: Homogenized Total")
        results['1g_total'].print_summary("  ")
        
        # 2-group by region
        print("\n--- 2-Group Cross Sections by Region ---")
        for region_idx, region in enumerate(regions):
            cell_indices = [c.index for c in self.geom.cells if c.region_index == region_idx]
            if cell_indices:
                name = f"2-Group: {region.material.name}"
                results[f'2g_region_{region_idx}'] = self.collapse_to_2group(
                    cell_indices=cell_indices, name=name
                )
                results[f'2g_region_{region_idx}'].print_summary("  ")
        
        # 2-group whole problem
        print("\n--- 2-Group Homogenized (Whole Problem) ---")
        results['2g_total'] = self.collapse_to_2group(name="2-Group: Homogenized Total")
        results['2g_total'].print_summary("  ")
        
        return results
    
    def export_to_dict(self, xs: CollapsedXS) -> dict:
        """Export CollapsedXS to a dictionary for serialization."""
        return {
            'name': xs.name,
            'n_groups': xs.n_groups,
            'volume': xs.volume,
            'sigma_t': xs.sigma_t.tolist(),
            'sigma_a': xs.sigma_a.tolist(),
            'sigma_s': xs.sigma_s.tolist(),
            'sigma_tr': xs.sigma_tr.tolist(),
            'sigma_f': xs.sigma_f.tolist(),
            'nu_sigma_f': xs.nu_sigma_f.tolist(),
            'kappa_sigma_f': xs.kappa_sigma_f.tolist(),
            'scatter_matrix': xs.scatter_matrix.tolist(),
            'chi': xs.chi.tolist(),
            'flux': xs.flux.tolist()
        }


def demonstrate_postprocessor():
    """Demonstrate the post-processor on a simple problem."""
    from mpact_reader import MPACTLibrary
    from mpact_material import Material
    
    print("="*70)
    print("CROSS-SECTION POST-PROCESSOR DEMONSTRATION")
    print("="*70)
    
    # Load library
    lib = MPACTLibrary('mpact8g_70s_v4_0m0_02232015.fmt')
    
    # Create a simple 2-region problem: UO2 + H2O
    print("\nCreating 2-region problem: UO2 fuel + H2O moderator")
    
    # UO2 fuel
    uo2 = Material("UO2 Fuel", temperature=900.0)
    u235 = lib.find_nuclide_by_name('U-235')
    u238 = lib.find_nuclide_by_name('U-238')
    o16 = lib.find_nuclide_by_name('O-16')
    N_uo2 = 0.0232
    uo2.add_nuclide(u235, N_uo2 * 0.05 / 3.0)
    uo2.add_nuclide(u238, N_uo2 * 0.95 / 3.0)
    uo2.add_nuclide(o16, N_uo2 * 2.0 / 3.0)
    
    # H2O moderator
    h2o = Material("H2O Moderator", temperature=600.0)
    h1 = lib.find_nuclide_by_name('H-1')
    N_h2o = 0.1003
    h2o.add_nuclide(h1, N_h2o * 2.0 / 3.0)
    h2o.add_nuclide(o16, N_h2o * 1.0 / 3.0)
    
    # Create geometry
    geom = OneDimensionalCartesianGeometryAndMesh("UO2 + H2O Slab")
    geom.add_region(uo2, length=1.0, n_cells=20, temperature=900.0)
    geom.add_region(h2o, length=1.0, n_cells=20, temperature=600.0)
    geom.finalize()
    
    # Solve transport
    print("\nSolving transport problem...")
    solver = LDFESNTransportSolver(geom, quadrature_order=8,
                                   left_bc='reflecting', right_bc='reflecting')
    solver.max_iterations = 5000
    solver.solve()
    solver.print_summary()
    
    # Post-process
    print("\n" + "="*70)
    print("POST-PROCESSING")
    print("="*70)
    
    pp = CrossSectionPostProcessor(solver)
    results = pp.process_all()
    
    return results


if __name__ == "__main__":
    results = demonstrate_postprocessor()
