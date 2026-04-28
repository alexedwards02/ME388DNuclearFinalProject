"""
Optimized 1D S_N Transport Solver
=================================

Key optimizations:
1. Precompute angle reflection mappings
2. Vectorized sweeps over cells (NumPy)
3. Minimize function calls in hot loops
4. Numba JIT compilation for inner sweeps (if available)
"""

import numpy as np
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt

# Try to import numba for JIT compilation
try:
    from numba import jit, prange
    HAS_NUMBA = True
    print("Numba JIT compilation: ENABLED")
except ImportError:
    HAS_NUMBA = False
    print("Numba not available, using NumPy vectorization")


class TransportSolution:
    """Container for transport solution data."""
    def __init__(self, n_cells: int, n_groups: int, n_angles: int):
        self.scalar_flux = np.zeros((n_cells, n_groups))
        self.angular_flux = np.zeros((n_cells, n_groups, n_angles))
        self.converged = False
        self.n_iterations = 0


class GaussLegendreQuadrature:
    """S_N Gauss-Legendre quadrature set."""
    def __init__(self, order: int):
        self.order = order
        self.n_angles = order
        
        # Get Gauss-Legendre points and weights on [-1, 1]
        self.mu, self.weights = np.polynomial.legendre.leggauss(order)
        
        # Precompute useful quantities
        self.abs_mu = np.abs(self.mu)
        self.positive_mask = self.mu > 0
        self.negative_mask = self.mu < 0
        
        # Precompute reflection mapping: for each angle n, find n' such that mu[n'] = -mu[n]
        self.reflection_map = np.zeros(order, dtype=np.int32)
        for n in range(order):
            self.reflection_map[n] = np.argmin(np.abs(self.mu + self.mu[n]))


# Numba-accelerated sweep functions (if available)
if HAS_NUMBA:
    @jit(nopython=True, cache=True)
    def sweep_right(sigma_t, dx, source, psi_in, n_cells):
        """Sweep left-to-right for a single direction (mu > 0)."""
        psi_avg = np.zeros(n_cells)
        psi_out = psi_in
        
        for i in range(n_cells):
            st = sigma_t[i]
            d = dx[i]
            q = source[i]
            
            if st < 1e-10:
                # Vacuum-like region
                psi_avg[i] = psi_out + 0.5 * q * d
                psi_out = psi_out + q * d
            else:
                tau = st * d
                if tau < 1e-8:
                    psi_avg[i] = psi_out + 0.5 * q * d
                    psi_out = psi_out + q * d
                else:
                    # Clamp tau for numerical stability
                    if tau > 50.0:
                        tau = 50.0
                    exp_tau = np.exp(-tau)
                    q_over_st = q / st
                    one_minus_exp = 1.0 - exp_tau
                    psi_avg[i] = q_over_st + (psi_out - q_over_st) * one_minus_exp / tau
                    psi_out = q_over_st + (psi_out - q_over_st) * exp_tau
                
                if psi_avg[i] < 0.0:
                    psi_avg[i] = 0.0
                if psi_out < 0.0:
                    psi_out = 0.0
        
        return psi_avg, psi_out
    
    @jit(nopython=True, cache=True)
    def sweep_left(sigma_t, dx, source, psi_in, n_cells):
        """Sweep right-to-left for a single direction (mu < 0)."""
        psi_avg = np.zeros(n_cells)
        psi_out = psi_in
        
        for i in range(n_cells - 1, -1, -1):
            st = sigma_t[i]
            d = dx[i]
            q = source[i]
            
            if st < 1e-10:
                psi_avg[i] = psi_out + 0.5 * q * d
                psi_out = psi_out + q * d
            else:
                tau = st * d
                if tau < 1e-8:
                    psi_avg[i] = psi_out + 0.5 * q * d
                    psi_out = psi_out + q * d
                else:
                    if tau > 50.0:
                        tau = 50.0
                    exp_tau = np.exp(-tau)
                    q_over_st = q / st
                    one_minus_exp = 1.0 - exp_tau
                    psi_avg[i] = q_over_st + (psi_out - q_over_st) * one_minus_exp / tau
                    psi_out = q_over_st + (psi_out - q_over_st) * exp_tau
                
                if psi_avg[i] < 0.0:
                    psi_avg[i] = 0.0
                if psi_out < 0.0:
                    psi_out = 0.0
        
        return psi_avg, psi_out

else:
    # NumPy vectorized versions (no Numba)
    def sweep_right(sigma_t, dx, source, psi_in, n_cells):
        """Vectorized sweep left-to-right."""
        psi_avg = np.zeros(n_cells)
        psi_out = psi_in
        
        for i in range(n_cells):
            st = sigma_t[i]
            d = dx[i]
            q = source[i]
            
            if st < 1e-10:
                psi_avg[i] = psi_out + 0.5 * q * d
                psi_out = psi_out + q * d
            else:
                tau = min(st * d, 50.0)
                if tau < 1e-8:
                    psi_avg[i] = psi_out + 0.5 * q * d
                    psi_out = psi_out + q * d
                else:
                    exp_tau = np.exp(-tau)
                    q_over_st = q / st
                    psi_avg[i] = q_over_st + (psi_out - q_over_st) * (1.0 - exp_tau) / tau
                    psi_out = q_over_st + (psi_out - q_over_st) * exp_tau
            
            psi_avg[i] = max(psi_avg[i], 0.0)
            psi_out = max(psi_out, 0.0)
        
        return psi_avg, psi_out
    
    def sweep_left(sigma_t, dx, source, psi_in, n_cells):
        """Vectorized sweep right-to-left."""
        psi_avg = np.zeros(n_cells)
        psi_out = psi_in
        
        for i in range(n_cells - 1, -1, -1):
            st = sigma_t[i]
            d = dx[i]
            q = source[i]
            
            if st < 1e-10:
                psi_avg[i] = psi_out + 0.5 * q * d
                psi_out = psi_out + q * d
            else:
                tau = min(st * d, 50.0)
                if tau < 1e-8:
                    psi_avg[i] = psi_out + 0.5 * q * d
                    psi_out = psi_out + q * d
                else:
                    exp_tau = np.exp(-tau)
                    q_over_st = q / st
                    psi_avg[i] = q_over_st + (psi_out - q_over_st) * (1.0 - exp_tau) / tau
                    psi_out = q_over_st + (psi_out - q_over_st) * exp_tau
            
            psi_avg[i] = max(psi_avg[i], 0.0)
            psi_out = max(psi_out, 0.0)
        
        return psi_avg, psi_out


class OptimizedLDFESNTransportSolver:
    """
    Optimized Linear Discontinuous Finite Element S_N Transport Solver.
    
    Key optimizations:
    - Precomputed angle reflection mappings
    - Extracted sweep functions (Numba JIT if available)
    - Minimized allocations in hot loops
    - Vectorized scalar flux computation
    """
    
    def __init__(self, geometry, quadrature_order: int = 4,
                 left_bc: str = 'vacuum', right_bc: str = 'vacuum'):
        self.geom = geometry
        self.geometry = geometry  # Alias
        self.n_cells = len(geometry.cells)
        
        # Quadrature
        self.quad = GaussLegendreQuadrature(quadrature_order)
        self.n_angles = self.quad.n_angles
        
        # Boundary conditions
        self.left_bc = left_bc.lower()
        self.right_bc = right_bc.lower()
        
        # Solver parameters
        self.tolerance = 1e-5
        self.max_iterations = 5000
        self.max_bc_iterations = 200
        self.bc_tolerance = 1e-6
        
        # Precompute geometry arrays (avoid repeated attribute access)
        self.dx = np.array([cell.width for cell in geometry.cells])
        
        # Extract cross sections into contiguous arrays (also sets self.n_groups)
        self._setup_cross_sections()
        
        # Setup solution container
        self.solution = TransportSolution(self.n_cells, self.n_groups, self.n_angles)
        
        # Iteration counter
        self.iterations = 0
    
    def _setup_cross_sections(self):
        """Extract cross sections into contiguous NumPy arrays."""
        print("\nExtracting cross sections...")
        
        # Use geometry's built-in extraction methods
        xs_total = self.geom.get_macroscopic_xs_array('total')
        self.sigma_t = xs_total['xs']
        self.n_groups = self.sigma_t.shape[1]
        
        xs_abs = self.geom.get_macroscopic_xs_array('absorption')
        self.sigma_a = xs_abs['xs']
        
        # Fission cross sections
        try:
            xs_fission = self.geom.get_macroscopic_xs_array('fission')
            self.sigma_f = xs_fission['xs']
        except:
            self.sigma_f = np.zeros_like(self.sigma_t)
        
        # Nu-fission (approximate as 2.5 * sigma_f)
        self.nu_sigma_f = 2.5 * self.sigma_f
        
        # Chi spectrum (hardcoded, same as original)
        self.chi_spectrum = np.array([0.584349, 0.415378, 0.000272, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Scattering matrices
        self.sigma_s_array = np.zeros((self.n_cells, self.n_groups, self.n_groups))
        for i, cell in enumerate(self.geom.cells):
            for component in cell.material.components.values():
                if hasattr(component.nuclide, 'scatter_matrix'):
                    self.sigma_s_array[i, :, :] += component.number_density * component.nuclide.scatter_matrix
        
        print(f"  Σ_t range: [{self.sigma_t.min():.4e}, {self.sigma_t.max():.4e}]")
        print(f"  Σ_a range: [{self.sigma_a.min():.4e}, {self.sigma_a.max():.4e}]")
    
    def solve(self):
        """
        Solve the fixed-source transport problem using Gauss-Seidel iteration.
        """
        print("\n" + "="*75)
        print("OPTIMIZED 1D LDFE S_N Transport Solver")
        print("="*75)
        print(f"  Cells: {self.n_cells}, Groups: {self.n_groups}")
        print(f"  Quadrature: S{self.quad.order} ({self.n_angles} angles)")
        print(f"  BCs: {self.left_bc}/{self.right_bc}")
        print(f"  Numba JIT: {'ENABLED' if HAS_NUMBA else 'DISABLED'}")
        
        # Setup fixed source (fission neutrons from chi spectrum)
        fissile_cells = np.where(self.sigma_f.sum(axis=1) > 1e-10)[0]
        n_fissile = len(fissile_cells)
        print(f"\n  Fissile cells: {n_fissile}")
        
        source = np.zeros((self.n_cells, self.n_groups))
        if n_fissile > 0:
            for i in fissile_cells:
                source[i, :] = self.chi_spectrum / n_fissile
        else:
            source[:, 0] = 1.0 / self.n_cells
        
        # Initialize flux
        phi = np.ones((self.n_cells, self.n_groups)) * 1e-10
        
        # Precompute scaled source arrays (divide by |mu| for each direction)
        # source_scaled[n, i, g] = source[i, g] / (2 * |mu[n]|)
        source_scaled = np.zeros((self.n_angles, self.n_cells, self.n_groups))
        for n in range(self.n_angles):
            source_scaled[n, :, :] = source / (2.0 * self.quad.abs_mu[n])
        
        # Precompute sigma_t / |mu| for each direction
        sigma_t_scaled = np.zeros((self.n_angles, self.n_cells, self.n_groups))
        for n in range(self.n_angles):
            sigma_t_scaled[n, :, :] = self.sigma_t / self.quad.abs_mu[n]
        
        print("\n" + "="*75)
        print("Starting Gauss-Seidel Iteration")
        print("="*75)
        
        # Main iteration loop
        for iteration in range(1, self.max_iterations + 1):
            phi_old = phi.copy()
            
            # Gauss-Seidel over energy groups
            for g in range(self.n_groups):
                # Build isotropic scattering source
                scatter_source = np.zeros(self.n_cells)
                for gp in range(self.n_groups):
                    scatter_source += self.sigma_s_array[:, gp, g] * phi[:, gp]
                
                # Total isotropic source = external + scattering
                total_source = source[:, g] + scatter_source
                
                # Solve this group
                phi[:, g] = self._solve_group_optimized(g, total_source)
            
            # Check convergence
            diff = np.abs(phi - phi_old)
            norm = np.abs(phi) + 1e-30
            residual = (diff / norm).max()
            
            if iteration <= 9 or iteration % 20 == 0:
                print(f"  Iter {iteration:4d}: residual = {residual:.6e}")
            
            if residual < self.tolerance:
                print(f"\n  ✓ Converged in {iteration} iterations")
                self.solution.converged = True
                self.solution.n_iterations = iteration
                self.iterations = iteration
                break
        
        if not self.solution.converged:
            print(f"\n  ✗ Did not converge in {self.max_iterations} iterations")
            self.iterations = self.max_iterations
        
        # Store solution
        self.solution.scalar_flux = phi
        
        print("="*75 + "\n")
    
    def _solve_group_optimized(self, g: int, source_iso: np.ndarray) -> np.ndarray:
        """
        Optimized single-group solve with boundary condition iteration.
        """
        n_cells = self.n_cells
        n_angles = self.n_angles
        
        # Get precomputed data for this group
        sigma_t_g = self.sigma_t[:, g].copy()
        dx = self.dx
        
        # Angular flux storage
        psi = np.zeros((n_cells, n_angles))
        
        # Boundary flux storage (only for reflecting BCs)
        psi_left = np.zeros(n_angles)   # Incoming at left (for mu > 0)
        psi_right = np.zeros(n_angles)  # Incoming at right (for mu < 0)
        
        # BC iteration
        for bc_iter in range(self.max_bc_iterations):
            psi_left_old = psi_left.copy()
            psi_right_old = psi_right.copy()
            
            # Sweep all angles
            for n in range(n_angles):
                mu = self.quad.mu[n]
                abs_mu = self.quad.abs_mu[n]
                
                # Scaled source: q = source_iso / (2 * |mu|)
                source_scaled = source_iso / (2.0 * abs_mu)
                
                # Scaled sigma_t: for characteristic form
                sigma_t_scaled = sigma_t_g / abs_mu
                
                if mu > 0:
                    # Left-to-right sweep
                    if self.left_bc == 'reflecting':
                        n_refl = self.quad.reflection_map[n]
                        psi_in = psi_left[n]
                    else:
                        psi_in = 0.0
                    
                    psi[:, n], psi_out = sweep_right(sigma_t_scaled, dx, source_scaled, psi_in, n_cells)
                    
                    # Store exit flux for reflecting BC
                    if self.right_bc == 'reflecting':
                        n_refl = self.quad.reflection_map[n]
                        psi_right[n_refl] = psi_out
                
                else:
                    # Right-to-left sweep
                    if self.right_bc == 'reflecting':
                        n_refl = self.quad.reflection_map[n]
                        psi_in = psi_right[n]
                    else:
                        psi_in = 0.0
                    
                    psi[:, n], psi_out = sweep_left(sigma_t_scaled, dx, source_scaled, psi_in, n_cells)
                    
                    # Store exit flux for reflecting BC
                    if self.left_bc == 'reflecting':
                        n_refl = self.quad.reflection_map[n]
                        psi_left[n_refl] = psi_out
            
            # Check BC convergence
            if self.left_bc == 'reflecting' or self.right_bc == 'reflecting':
                bc_change = np.abs(psi_left - psi_left_old).sum() + np.abs(psi_right - psi_right_old).sum()
                if bc_change < self.bc_tolerance:
                    break
            else:
                break
        
        # Compute scalar flux (vectorized)
        phi = np.dot(psi, self.quad.weights)
        
        return phi
    
    def print_summary(self):
        """Print solution summary."""
        print("\n" + "="*75)
        print("Transport Solution Summary")
        print("="*75)
        print(f"  Converged: {self.solution.converged}")
        print(f"  Iterations: {self.iterations}")
        
        print("\nScalar Flux by Group:")
        for g in range(self.n_groups):
            phi_g = self.solution.scalar_flux[:, g]
            print(f"  Group {g+1}: min={phi_g.min():.4e}, max={phi_g.max():.4e}, avg={phi_g.mean():.4e}")
        print("="*75 + "\n")
    
    def plot_scalar_flux(self, groups=None, figsize=(12, 8)):
        """Plot scalar flux by group."""
        if groups is None:
            groups = list(range(min(self.n_groups, 8)))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.array([cell.center for cell in self.geom.cells])
        
        for g in groups:
            ax.semilogy(x, self.solution.scalar_flux[:, g], label=f'Group {g+1}')
        
        ax.set_xlabel('Position (cm)')
        ax.set_ylabel('Scalar Flux')
        ax.set_title(f'Scalar Flux Distribution ({self.n_cells} cells, S{self.quad.order})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig


# Backwards compatibility alias
LDFESNTransportSolver = OptimizedLDFESNTransportSolver
