#!/usr/bin/env python
# coding: utf-8

# # ME388D Homework 3
# ### Question 3: Criticality with Sn
# 
# Please read the following notes.<br><br>
# 
# **You will need to download the file "mpact_data_8g.fmt" from Canvas and place it in the same folder as this notebook for the code to work.** <br><br>
# 
# This notebook contains an Sn transport solver that generates collapsed 1-group cross-sections. 
# You may examine the code if you want to see what's going on. 
# Scroll all the way to the bottom of this notebook for the student question.<br><br>
# 
# Homework 3, like most (but not all) homeworks, was written partially by Dr. Clarno and partially by the TA (Alex Macris). <br>
# Although this question was written by the TA, the code that used in the question was provided by Dr. Clarno and modifications were made for the purpose of this question.<br> 
# This code exists as a replacement for SCALE and this is the first time it has ever been given out to students. <br>
# If you have problems getting this notebook to work, please send an email to both Dr. Clarno (clarno@utexas.edu) and Alex Macris (macris@utexas.edu) to get it resolved as we may need to work together to fix it.<br> 
# (Of course, we have tested this notebook before distributing it and it should work.)

# In[1]:


"""MPACT Formatted Library Reader
================================
Reader for MPACT 8-group formatted cross section libraries.

Format: ASCII text with structured sections
- %VER: Version info
- %DIM: Dimensions
- %GRP: Energy group boundaries
- %CHI: Fission spectrum
- %DIR: Nuclide directory
- %NUC: Individual nuclide data with full scattering matrices
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class MPACTNuclide:
    """
    Container for MPACT nuclide data.

    Attributes:
        id: Nuclide ID (ZZAAA format)
        name: Nuclide name
        atomic_mass: Atomic mass (amu)
        ngroups: Number of energy groups
        temperatures: Available temperatures (K)
        cross_sections: Dictionary of cross sections
        scatter_matrix: Full scattering matrix [ngroups, ngroups] at reference temperature
    """
    id: int
    name: str = ""
    atomic_mass: float = 0.0
    ngroups: int = 8
    temperatures: np.ndarray = field(default_factory=lambda: np.array([]))
    cross_sections: Dict[str, np.ndarray] = field(default_factory=dict)
    scatter_matrix: np.ndarray = field(default_factory=lambda: np.array([]))

    def __repr__(self):
        return f"MPACTNuclide(id={self.id}, name='{self.name}', ngroups={self.ngroups})"


class MPACTLibrary:
    """
    Reader for MPACT formatted cross section libraries.
    """

    def __init__(self, filename: str = None):
        """
        Initialize MPACT library reader.

        Args:
            filename: Path to MPACT library file (.fmt)
        """
        self.filename = filename
        self.version = None
        self.nuclides: Dict[int, MPACTNuclide] = {}
        self.energy_bounds = None
        self.chi_spectrum = None
        self.ngroups = 8

        if filename:
            self.load(filename)

    def load(self, filename: str):
        """Load MPACT library from file."""
        self.filename = Path(filename)
        if not self.filename.exists():
            raise FileNotFoundError(f"File not found: {filename}")

        print(f"Loading MPACT library: {filename}")

        with open(self.filename, 'r') as f:
            lines = [line.strip() for line in f.readlines()]

        self._parse_header(lines)
        self._parse_nuclides(lines)

        print(f"Successfully loaded {len(self.nuclides)} nuclides")
        print(f"  Energy groups: {self.ngroups}")
        print(f"  Energy bounds: {self.energy_bounds[0]:.2e} eV to {self.energy_bounds[-1]:.2e} eV")

    def _parse_header(self, lines: List[str]):
        """Parse header sections."""
        i = 0
        while i < len(lines):
            line = lines[i]

            if line.startswith('%VER'):
                # Version
                i += 1
                self.version = lines[i].strip()

            elif line.startswith('%DIM'):
                # Dimensions
                i += 1
                dims = [int(x) for x in lines[i].split()]
                self.ngroups = dims[0]

            elif line.startswith('%GRP'):
                # Energy group boundaries (upper bounds)
                i += 1
                self.energy_bounds = np.array([float(x) for x in lines[i].split()])
                # Note: MPACT gives upper bounds, we'll prepend 0 for lower bound of last group
                self.energy_bounds = np.append(self.energy_bounds, 1.0e-5)

            elif line.startswith('%CHI'):
                # Fission spectrum
                i += 1
                self.chi_spectrum = np.array([float(x) for x in lines[i].split()])

            elif line.startswith('%DIR'):
                # End of header
                break

            i += 1


    def _parse_nuclides(self, lines: List[str]):
        """Parse nuclide data sections."""
        i = 0
        while i < len(lines):
            if lines[i].startswith('%NUC'):
                # Parse this nuclide
                nuc_line = lines[i].split()
                nuc_id = int(nuc_line[2])
                nuc_name = nuc_line[-1]
                atomic_mass = float(nuc_line[3])

                # Find temperature line
                i += 1
                while i < len(lines) and not lines[i].startswith('TP1+'):
                    i += 1

                if i < len(lines):
                    i += 1
                    temps = np.array([float(x) for x in lines[i].split()])
                    n_temps = len(temps)

                    # Find XSD+ section
                    while i < len(lines) and not lines[i].startswith('XSD+'):
                        i += 1

                    if i < len(lines):
                        xsd_start = i + 1

                        # Find RIA+ section (if exists)
                        ria_start = None
                        j = i
                        while j < len(lines) and not lines[j].startswith('%NUC'):
                            if lines[j].startswith('RIA+'):
                                ria_start = j + 1
                                break
                            j += 1

                        # Parse cross section data
                        nuc = self._parse_xs_data(lines, xsd_start, nuc_id, nuc_name, 
                                                  atomic_mass, temps, ria_start)
                        self.nuclides[nuc_id] = nuc

            i += 1

    def _parse_xs_data(self, lines: List[str], start_idx: int, 
                      nuc_id: int, nuc_name: str, atomic_mass: float,
                      temps: np.ndarray, ria_start_idx: Optional[int] = None) -> MPACTNuclide:
        """
        Parse XSD+ cross section data for one nuclide.

        XSD+ format:
        group temp_idx sigma_tr sigma_f nu_sigma_f sigma_total scatter_total start_group end_group [scatter_xs...]

        CORRECT column interpretation:
        - parts[2]: Transport cross section (σ_tr)
        - parts[3]: Fission cross section (σ_f) - NOT absorption!
        - parts[4]: nu*sigma_f
        - parts[5]: Total cross section (σ_t)
        - parts[6]: Total scattering cross section (σ_s)
        - parts[7-8]: start_group, end_group (1-indexed)
        - parts[9+]: Group-to-group scattering

        Derived quantities:
        - σ_a (absorption) = σ_t - σ_s = σ_f + σ_c (capture)
        - σ_c (capture) = σ_a - σ_f
        """
        nuc = MPACTNuclide(
            id=nuc_id,
            name=nuc_name,
            atomic_mass=atomic_mass,
            ngroups=self.ngroups,
            temperatures=temps
        )

        n_temps = len(temps)

        # Initialize arrays for all temperatures
        sigma_t = np.zeros(self.ngroups)
        sigma_f = np.zeros(self.ngroups)  # Fission, not absorption!
        sigma_a = np.zeros(self.ngroups)  # Will be calculated as sigma_t - sigma_s
        nu_sigma_f = np.zeros(self.ngroups)
        sigma_s_total = np.zeros(self.ngroups)
        scatter_matrix = np.zeros((self.ngroups, self.ngroups))

        # Also store temperature-dependent data
        sigma_t_all_temps = np.zeros((n_temps, self.ngroups))
        sigma_f_all_temps = np.zeros((n_temps, self.ngroups))
        sigma_a_all_temps = np.zeros((n_temps, self.ngroups))
        nu_sigma_f_all_temps = np.zeros((n_temps, self.ngroups))
        sigma_s_total_all_temps = np.zeros((n_temps, self.ngroups))

        # Parse XSD+ lines
        i = start_idx
        while i < len(lines):
            line = lines[i]

            # Check for end of XSD+ section
            if line.startswith('%') or line.startswith('SP1+'):
                break

            if not line:
                i += 1
                continue

            parts = line.split()
            if len(parts) < 9:  # Need at least up to end_group
                i += 1
                continue

            try:
                group = int(parts[0]) - 1  # Convert to 0-indexed
                temp_idx = int(parts[1]) - 1

                # Store data for all temperatures
                if 0 <= temp_idx < n_temps and group < self.ngroups:
                    sigma_t_all_temps[temp_idx, group] = float(parts[5])
                    sigma_f_all_temps[temp_idx, group] = float(parts[3])  # parts[3] is FISSION!
                    nu_sigma_f_all_temps[temp_idx, group] = float(parts[4])
                    sigma_s_total_all_temps[temp_idx, group] = float(parts[6])
                    # Calculate absorption as total - scatter (will be overridden by RIA+ if available)
                    calculated_abs = float(parts[5]) - float(parts[6])
                    # For fast groups where this is negative, use fission only (capture negligible)
                    if calculated_abs > 0:
                        sigma_a_all_temps[temp_idx, group] = calculated_abs
                    else:
                        sigma_a_all_temps[temp_idx, group] = float(parts[3])  # Use fission only

                # Only process reference temperature for scatter matrix
                if temp_idx == 0 and group < self.ngroups:
                    sigma_t[group] = float(parts[5])
                    sigma_f[group] = float(parts[3])  # parts[3] is FISSION!
                    nu_sigma_f[group] = float(parts[4])
                    sigma_s_total[group] = float(parts[6])
                    # Calculate absorption as total - scatter (will be overridden by RIA+ if available)
                    calculated_abs = float(parts[5]) - float(parts[6])
                    # For fast groups where this is negative, use fission only
                    if calculated_abs > 0:
                        sigma_a[group] = calculated_abs
                    else:
                        sigma_a[group] = float(parts[3])  # Use fission only

                    # Parse scattering matrix
                    # Format: start_group end_group [scatter_xs...]
                    start_group = int(parts[7]) - 1  # Convert to 0-indexed
                    end_group = int(parts[8]) - 1    # Convert to 0-indexed
                    n_scatter = end_group - start_group + 1

                    # Extract scattering cross sections
                    # Format: Scattering TO current group FROM groups [start_group, ..., end_group]
                    scatter_xs_start = 9
                    for j in range(n_scatter):
                        if scatter_xs_start + j < len(parts):
                            g_from = start_group + j
                            if 0 <= g_from < self.ngroups:
                                # Store as scatter_matrix[from, to]
                                scatter_matrix[g_from, group] = float(parts[scatter_xs_start + j])

            except (ValueError, IndexError):
                pass

            i += 1

        # Parse RIA+ section if available (Resonance Integral Absorption)
        # This provides absorption at infinite dilution for resonance materials
        if ria_start_idx is not None:
            self._parse_ria_data(lines, ria_start_idx, sigma_a_all_temps, n_temps)
            # Update reference temperature absorption
            sigma_a = sigma_a_all_temps[0, :]

        # Store cross sections at reference temperature
        nuc.cross_sections['total'] = sigma_t
        nuc.cross_sections['absorption'] = sigma_a  # sigma_a = sigma_t - sigma_s
        nuc.cross_sections['fission'] = sigma_f
        nuc.cross_sections['capture'] = sigma_a - sigma_f  # sigma_c = sigma_a - sigma_f
        nuc.cross_sections['nu-fission'] = nu_sigma_f
        nuc.cross_sections['scatter_total'] = sigma_s_total
        nuc.scatter_matrix = scatter_matrix

        # Store temperature-dependent cross sections
        nuc.cross_sections['total_all_temps'] = sigma_t_all_temps
        nuc.cross_sections['absorption_all_temps'] = sigma_a_all_temps
        nuc.cross_sections['fission_all_temps'] = sigma_f_all_temps
        nuc.cross_sections['capture_all_temps'] = sigma_a_all_temps - sigma_f_all_temps
        nuc.cross_sections['nu-fission_all_temps'] = nu_sigma_f_all_temps
        nuc.cross_sections['scatter_total_all_temps'] = sigma_s_total_all_temps

        return nuc

    def _parse_ria_data(self, lines: List[str], start_idx: int,
                       sigma_a_all_temps: np.ndarray, n_temps: int):
        """
        Parse RIA+ (Resonance Integral Absorption) section.

        Format: group temp_idx absorption_at_sigma0[0] ... absorption_at_sigma0[15]

        We use the last value (index 15) which corresponds to σ₀ → ∞ (infinite dilution).
        This gives the true absorption cross section including capture for resonance materials.

        Only updates groups that have RIA+ data (typically resonance groups).
        For other groups, keeps the calculated value from XSD+ data.

        Args:
            lines: File lines
            start_idx: Start index of RIA+ data
            sigma_a_all_temps: Array to update with RIA+ absorption data
            n_temps: Number of temperatures
        """
        i = start_idx
        while i < len(lines):
            line = lines[i]

            # Check for end of RIA+ section
            if line.startswith('%') or line.strip().startswith('RIS+') or line.strip().startswith('SA'):
                break

            if not line.strip():
                i += 1
                continue

            parts = line.split()
            if len(parts) < 18:  # Need group, temp_idx, and 16 absorption values
                i += 1
                continue

            try:
                group = int(parts[0]) - 1  # Convert to 0-indexed
                temp_idx = int(parts[1]) - 1

                # Extract absorption at infinite dilution (last value, index 17 = parts[1] + 16)
                if 0 <= temp_idx < n_temps and group < self.ngroups and len(parts) >= 18:
                    sigma_a_inf = float(parts[17])  # Last value = infinite dilution
                    # Only update if positive (RIA+ data exists for this group)
                    if sigma_a_inf > 0:
                        sigma_a_all_temps[temp_idx, group] = sigma_a_inf

            except (ValueError, IndexError):
                pass

            i += 1

    def get_nuclide(self, nuclide_id: int) -> MPACTNuclide:
        """Get nuclide by ID."""
        if nuclide_id not in self.nuclides:
            raise ValueError(f"Nuclide {nuclide_id} not found in library")
        return self.nuclides[nuclide_id]

    def list_nuclides(self) -> List[tuple]:
        """List all available nuclides."""
        return [(nuc_id, nuc.name) for nuc_id, nuc in sorted(self.nuclides.items())]

    def find_nuclide_by_name(self, name: str) -> Optional[MPACTNuclide]:
        """Find nuclide by name (case-insensitive)."""
        name_upper = name.upper()
        for nuc in self.nuclides.values():
            if nuc.name.upper() == name_upper:
                return nuc
        return None



# In[2]:


"""1D S_N Transport Solver
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



    def solve(self):
        """
        Solve the fixed-source transport problem using Gauss-Seidel iteration.
        """


        # Setup fixed source (fission neutrons from chi spectrum)
        fissile_cells = np.where(self.sigma_f.sum(axis=1) > 1e-10)[0]
        n_fissile = len(fissile_cells)

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

        # for g in groups:
        print(sum(self.solution.scalar_flux[:, g] for g in groups))
        ax.semilogy(x, sum(self.solution.scalar_flux[:, g] for g in groups))#, label=f'Group {g+1}')

        ax.set_xlabel('Position (cm)')
        ax.set_ylabel('Scalar Flux')
        ax.set_title(f'Scalar Flux Distribution ({self.n_cells} cells, S{self.quad.order})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig


# Backwards compatibility alias
LDFESNTransportSolver = OptimizedLDFESNTransportSolver


# In[3]:


"""Material Class for MPACT Library
=================================

Extension to the MPACT reader that provides a Material class for managing
mixtures of nuclides with number densities and computing macroscopic cross sections.

Macroscopic cross section: Σ = Σ(N_i * σ_i) where N_i is number density [atoms/barn-cm]
                                               and σ_i is microscopic xs [barns]
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class MaterialComponent:
    """
    Container for a single nuclide component in a material.

    Attributes:
        nuclide: MPACTNuclide object containing cross-section data
        number_density: Number density in atoms/(barn-cm)
        weight_fraction: Weight fraction (optional, for reference)
    """
    nuclide: MPACTNuclide
    number_density: float  # atoms/(barn-cm)
    weight_fraction: Optional[float] = None

    def __repr__(self):
        return (f"MaterialComponent(nuclide={self.nuclide.name}, "
                f"N={self.number_density:.6e} atoms/barn-cm)")


class Material:
    """
    Represents a homogeneous material mixture of multiple nuclides.

    A material consists of one or more nuclides, each with an associated
    number density. The class provides methods to compute macroscopic
    cross sections by summing contributions from all nuclides.

    Attributes:
        name: Material name/identifier
        components: Dictionary of MaterialComponent objects keyed by nuclide ID
        temperature: Material temperature in Kelvin (for reference)
        density: Material mass density in g/cm³ (for reference)
        metadata: Additional material properties
    """

    def __init__(self, name: str = "Material", temperature: float = 293.6,
                 density: Optional[float] = None):
        """
        Initialize a new Material.

        Args:
            name: Material name
            temperature: Temperature in Kelvin (default: 293.6 K = 20.45°C)
            density: Mass density in g/cm³ (optional)

        Example:
            >>> mat = Material("UO2 Fuel", temperature=900.0, density=10.4)
        """
        self.name = name
        self.temperature = temperature
        self.density = density
        self.components: Dict[int, MaterialComponent] = {}
        self.metadata = {}

    def add_nuclide(self, nuclide: MPACTNuclide, number_density: float,
                    weight_fraction: Optional[float] = None):
        """
        Add a nuclide to the material with specified number density.

        Args:
            nuclide: MPACTNuclide object from MPACT library
            number_density: Number density in atoms/(barn-cm)
            weight_fraction: Weight fraction (optional, for reference)

        Example:
            >>> u235 = lib.find_nuclide_by_name('U-235')
            >>> mat.add_nuclide(u235, 1.5e-3, weight_fraction=0.05)
        """
        if not hasattr(nuclide, 'cross_sections'):
            raise TypeError("nuclide must have cross_sections attribute (MPACTNuclide)")

        if number_density <= 0:
            raise ValueError("Number density must be positive")

        component = MaterialComponent(
            nuclide=nuclide,
            number_density=number_density,
            weight_fraction=weight_fraction
        )

        self.components[nuclide.id] = component

    def remove_nuclide(self, nuclide_id: int):
        """Remove a nuclide from the material."""
        if nuclide_id in self.components:
            del self.components[nuclide_id]
        else:
            raise KeyError(f"Nuclide {nuclide_id} not found in material")

    def get_number_density(self, nuclide_id: int) -> float:
        """Get number density for a specific nuclide."""
        if nuclide_id not in self.components:
            raise KeyError(f"Nuclide {nuclide_id} not in material")
        return self.components[nuclide_id].number_density

    def set_number_density(self, nuclide_id: int, number_density: float):
        """Update number density for a nuclide."""
        if nuclide_id not in self.components:
            raise KeyError(f"Nuclide {nuclide_id} not in material")
        if number_density <= 0:
            raise ValueError("Number density must be positive")
        self.components[nuclide_id].number_density = number_density

    def get_total_number_density(self) -> float:
        """Calculate total atomic number density of the material."""
        return sum(comp.number_density for comp in self.components.values())

    def get_macroscopic_xs(self, reaction: str = 'total') -> np.ndarray:
        """
        Compute macroscopic cross section for a specific reaction.

        The macroscopic cross section is calculated as:
            Σ(E) = Σᵢ Nᵢ * σᵢ(E)

        where:
            Σ(E) = macroscopic cross section [cm⁻¹]
            Nᵢ = number density of nuclide i [atoms/(barn-cm)]
            σᵢ(E) = microscopic cross section of nuclide i [barns]

        Note: Computes TRUE total as: Σ_t = Σ_abs + Σ_scatter

        Args:
            reaction: Reaction type ('total', 'absorption', 'fission', 'capture', etc.)

        Returns:
            NumPy array of macroscopic cross sections by energy group [cm⁻¹]
        """
        if not self.components:
            raise ValueError("Material has no nuclides")

        # Determine energy group structure from first nuclide
        first_component = next(iter(self.components.values()))
        ngroups = first_component.nuclide.ngroups

        # Special handling for 'total' - compute as absorption + scattering
        if reaction == 'total':
            macro_abs = np.zeros(ngroups)
            for component in self.components.values():
                micro_abs = component.nuclide.cross_sections.get('absorption')
                if micro_abs is not None:
                    macro_abs += component.number_density * micro_abs

            macro_scatter = np.zeros(ngroups)
            for component in self.components.values():
                nuclide = component.nuclide
                N = component.number_density

                micro_scatter = nuclide.cross_sections.get('scatter_total')
                if micro_scatter is not None:
                    macro_scatter += N * micro_scatter
                elif hasattr(nuclide, 'scatter_matrix'):
                    if nuclide.scatter_matrix.shape == (ngroups, ngroups):
                        macro_scatter += N * nuclide.scatter_matrix.sum(axis=1)

            return macro_abs + macro_scatter

        # For all other reactions, compute directly
        macro_xs = np.zeros(ngroups)
        for component in self.components.values():
            micro_xs = component.nuclide.cross_sections.get(reaction)
            if micro_xs is not None:
                macro_xs += component.number_density * micro_xs

        return macro_xs

    def get_all_macroscopic_xs(self) -> Dict[str, np.ndarray]:
        """Compute macroscopic cross sections for all available reactions."""
        first_component = next(iter(self.components.values()))
        available_reactions = list(first_component.nuclide.cross_sections.keys())

        macro_xs_dict = {}
        for reaction in available_reactions:
            if '_all_temps' in reaction:
                continue
            try:
                macro_xs_dict[reaction] = self.get_macroscopic_xs(reaction)
            except:
                pass

        if 'total' not in macro_xs_dict:
            macro_xs_dict['total'] = self.get_macroscopic_xs('total')

        return macro_xs_dict

    def get_nuclide_contributions(self, reaction: str = 'total') -> Dict[int, np.ndarray]:
        """Get individual nuclide contributions to macroscopic cross section."""
        contributions = {}

        for nuc_id, component in self.components.items():
            if reaction == 'total':
                micro_abs = component.nuclide.cross_sections.get('absorption', 0)
                micro_scatter = component.nuclide.cross_sections.get('scatter_total', 0)
                contrib = component.number_density * (micro_abs + micro_scatter)
            else:
                micro_xs = component.nuclide.cross_sections.get(reaction)
                if micro_xs is not None:
                    contrib = component.number_density * micro_xs
                else:
                    continue

            contributions[nuc_id] = contrib

        return contributions

    def plot_macroscopic_xs(self, reactions: Optional[List[str]] = None,
                           logy: bool = True, figsize: Tuple[float, float] = (12, 7)):
        """Plot macroscopic cross sections for specified reactions."""
        if reactions is None:
            reactions = ['total', 'absorption', 'fission', 'scatter_total']

        fig, ax = plt.subplots(figsize=figsize)

        first_component = next(iter(self.components.values()))
        energy_groups = np.arange(1, first_component.nuclide.ngroups + 1)

        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
        for i, reaction in enumerate(reactions):
            try:
                macro_xs = self.get_macroscopic_xs(reaction)
                marker = markers[i % len(markers)]
                ax.plot(energy_groups, macro_xs, marker=marker, linestyle='-',
                       label=reaction.replace('_', ' ').title(),
                       linewidth=2.5, markersize=7, alpha=0.8)
            except:
                pass

        ax.set_xlabel('Energy Group', fontsize=13, fontweight='bold')
        ax.set_ylabel('Macroscopic Cross Section Σ (cm⁻¹)', fontsize=13, fontweight='bold')
        ax.set_title(f'Macroscopic Cross Sections: {self.name}', fontsize=15, fontweight='bold')
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_xticks(energy_groups)

        if logy:
            ax.set_yscale('log')
            ax.set_ylim(bottom=1e-4)

        plt.tight_layout()
        return fig, ax

    def plot_nuclide_contributions(self, reaction: str = 'total', logy: bool = True,
                                   figsize: Tuple[float, float] = (12, 7)):
        """Plot individual nuclide contributions to macroscopic cross section."""
        contributions = self.get_nuclide_contributions(reaction)

        if not contributions:
            raise ValueError(f"No contributions found for reaction: {reaction}")

        fig, ax = plt.subplots(figsize=figsize)

        first_contrib = next(iter(contributions.values()))
        energy_groups = np.arange(1, len(first_contrib) + 1)

        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        for i, (nuc_id, contrib) in enumerate(sorted(contributions.items())):
            nuclide = self.components[nuc_id].nuclide
            marker = markers[i % len(markers)]
            ax.plot(energy_groups, contrib, marker=marker, linestyle='-',
                   label=f'{nuclide.name}', linewidth=2, markersize=6, alpha=0.8)

        total_macro_xs = self.get_macroscopic_xs(reaction)
        ax.plot(energy_groups, total_macro_xs, marker='s', linestyle='--',
               label='Total', linewidth=3, markersize=8, color='black')

        ax.set_xlabel('Energy Group', fontsize=13, fontweight='bold')
        ax.set_ylabel(f'{reaction.capitalize()} Σ (cm⁻¹)', fontsize=13, fontweight='bold')
        ax.set_title(f'Nuclide Contributions to {reaction.capitalize()} Σ: {self.name}',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10, framealpha=0.9, ncol=2)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_xticks(energy_groups)

        if logy:
            ax.set_yscale('log')
            ax.set_ylim(bottom=1e-5)

        plt.tight_layout()
        return fig, ax

    def summary(self):
        """Print detailed summary of material composition and properties."""
        print("\n" + "="*75)
        print(f"Material Summary: {self.name}")
        print("="*75)
        print(f"Temperature:     {self.temperature:.2f} K")
        if self.density is not None:
            print(f"Mass Density:    {self.density:.4f} g/cm³")

        total_N = self.get_total_number_density()
        print(f"Total Number Density: {total_N:.6e} atoms/(barn-cm)")
        print(f"Number of Nuclides:   {len(self.components)}")

        print("\nComposition:")
        print("-"*75)
        print(f"{'Nuclide ID':<12} {'Name':<15} {'Number Density':<20} {'Atom %':<10}")
        print("-"*75)

        for nuc_id, component in sorted(self.components.items()):
            atom_percent = 100.0 * component.number_density / total_N
            print(f"{nuc_id:<12} {component.nuclide.name:<15} "
                  f"{component.number_density:<20.6e} {atom_percent:<10.4f}")

        print("\nKey Macroscopic Cross Sections at Reference Temperature:")
        print("-"*75)
        try:
            reactions_to_show = ['total', 'absorption', 'fission', 'capture', 'scatter_total']
            for reaction in reactions_to_show:
                try:
                    macro_xs = self.get_macroscopic_xs(reaction)
                    print(f"  {reaction:<15} : max Σ = {macro_xs.max():.6e} cm⁻¹, "
                          f"min Σ = {macro_xs.min():.6e} cm⁻¹")
                except:
                    pass
        except Exception as e:
            print(f"  Could not compute macroscopic cross sections: {e}")

        print("="*75 + "\n")

    def __repr__(self):
        return (f"Material(name='{self.name}', nuclides={len(self.components)}, "
                f"N_total={self.get_total_number_density():.4e} atoms/barn-cm)")

    def __len__(self):
        return len(self.components)


# ============================================================================
# Utility Functions
# ============================================================================

def create_material_from_atom_fractions(name: str, library: MPACTLibrary,
                                       atom_fractions: Dict[int, float],
                                       total_number_density: float,
                                       temperature: float = 293.6,
                                       density: Optional[float] = None) -> Material:
    """Create a Material from atomic fractions."""
    total_fraction = sum(atom_fractions.values())
    if abs(total_fraction - 1.0) > 1e-6:
        print(f"Warning: Atom fractions sum to {total_fraction}, normalizing...")
        atom_fractions = {k: v/total_fraction for k, v in atom_fractions.items()}

    material = Material(name, temperature=temperature, density=density)

    for nuc_id, fraction in atom_fractions.items():
        nuclide = library.get_nuclide(nuc_id)
        if nuclide is None:
            raise ValueError(f"Nuclide {nuc_id} not found in library")

        number_density = fraction * total_number_density
        material.add_nuclide(nuclide, number_density, weight_fraction=None)

    return material


def compare_materials(materials: List[Material], reaction: str = 'total',
                     logy: bool = True, figsize: Tuple[float, float] = (14, 8)):
    """Compare macroscopic cross sections for multiple materials."""
    if not materials:
        raise ValueError("No materials to compare")

    fig, ax = plt.subplots(figsize=figsize)

    first_xs = materials[0].get_macroscopic_xs(reaction)
    energy_groups = np.arange(1, len(first_xs) + 1)

    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    for i, material in enumerate(materials):
        macro_xs = material.get_macroscopic_xs(reaction)
        marker = markers[i % len(markers)]
        ax.plot(energy_groups, macro_xs, marker=marker, linestyle='-',
               label=material.name, linewidth=2.5, markersize=7, alpha=0.8)

    ax.set_xlabel('Energy Group', fontsize=13, fontweight='bold')
    ax.set_ylabel(f'{reaction.capitalize()} Σ (cm⁻¹)', fontsize=13, fontweight='bold')
    ax.set_title(f'Material Comparison: {reaction.capitalize()} Cross Sections',
                fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xticks(energy_groups)

    if logy:
        ax.set_yscale('log')
        ax.set_ylim(bottom=1e-4)

    plt.tight_layout()
    return fig, ax


# In[4]:


"""One-Dimensional Cartesian Geometry and Mesh
============================================

Provides a 1D Cartesian geometry system for reactor physics calculations
with material regions, mesh discretization, and cross-section management.

Features:
- Material region definition with lengths and temperatures
- Automatic mesh generation with specified cell counts per region
- Node placement at material interfaces
- Cross-section mapping to mesh cells
- Visualization of geometry and properties
- Compatible with Linear Discontinuous Finite Element Methods
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class MeshCell:
    """
    Represents a single mesh cell in the 1D geometry.

    Attributes:
        index: Global cell index (0-based)
        region_index: Index of the region this cell belongs to
        left_node: Left boundary position [cm]
        right_node: Right boundary position [cm]
        center: Cell center position [cm]
        width: Cell width [cm]
        material: Material object for this cell
        temperature: Temperature in Kelvin
    """
    index: int
    region_index: int
    left_node: float
    right_node: float
    center: float
    width: float
    material: Material
    temperature: float

    def __repr__(self):
        return (f"MeshCell(idx={self.index}, region={self.region_index}, "
                f"x=[{self.left_node:.4f}, {self.right_node:.4f}] cm)")


@dataclass
class MaterialRegion:
    """
    Represents a material region in the 1D geometry.

    Attributes:
        index: Region index (0-based)
        material: Material object
        length: Region length [cm]
        n_cells: Number of mesh cells in this region
        temperature: Temperature in Kelvin
        left_boundary: Left boundary position [cm]
        right_boundary: Right boundary position [cm]
        cell_width: Width of each cell [cm]
    """
    index: int
    material: Material
    length: float
    n_cells: int
    temperature: float
    left_boundary: float = 0.0
    right_boundary: float = 0.0
    cell_width: float = 0.0

    def __post_init__(self):
        """Calculate derived quantities."""
        self.right_boundary = self.left_boundary + self.length
        self.cell_width = self.length / self.n_cells
        # Update material temperature
        self.material.temperature = self.temperature

    def __repr__(self):
        return (f"MaterialRegion(idx={self.index}, material='{self.material.name}', "
                f"length={self.length:.4f} cm, n_cells={self.n_cells}, T={self.temperature:.1f} K)")


class OneDimensionalCartesianGeometryAndMesh:
    """
    One-dimensional Cartesian geometry with material regions and mesh.

    This class manages a 1D reactor geometry consisting of multiple material
    regions, each discretized into uniform mesh cells. Nodes are placed at
    material interfaces to ensure proper boundary treatment.

    Attributes:
        regions: List of MaterialRegion objects
        cells: List of MeshCell objects
        nodes: Array of node positions [cm]
        total_length: Total geometry length [cm]
        n_cells_total: Total number of mesh cells
        n_nodes: Total number of nodes
    """

    def __init__(self, name: str = "1D Geometry"):
        """
        Initialize empty geometry.

        Args:
            name: Geometry name/identifier
        """
        self.name = name
        self.regions: List[MaterialRegion] = []
        self.cells: List[MeshCell] = []
        self.nodes: np.ndarray = np.array([])
        self.total_length = 0.0
        self.n_cells_total = 0
        self.n_nodes = 0
        self._finalized = False

    def add_region(self, material: Material, length: float, n_cells: int,
                   temperature: Optional[float] = None):
        """
        Add a material region to the geometry.

        Regions are added sequentially from left to right.

        Args:
            material: Material object for this region
            length: Region length in cm (must be > 0)
            n_cells: Number of mesh cells (must be >= 1)
            temperature: Temperature in Kelvin (if None, uses material.temperature)

        Example:
            >>> geom = OneDimensionalCartesianGeometryAndMesh("Test")
            >>> geom.add_region(fuel_material, length=2.0, n_cells=20, temperature=900.0)
            >>> geom.add_region(water_material, length=1.0, n_cells=10, temperature=350.0)
        """
        if self._finalized:
            raise RuntimeError("Cannot add regions after geometry is finalized")

        if length <= 0:
            raise ValueError(f"Region length must be positive, got {length}")

        if n_cells < 1:
            raise ValueError(f"Number of cells must be >= 1, got {n_cells}")

        # Use material temperature if not specified
        if temperature is None:
            temperature = material.temperature

        # Calculate region boundaries
        left_boundary = self.total_length

        # Create region
        region = MaterialRegion(
            index=len(self.regions),
            material=material,
            length=length,
            n_cells=n_cells,
            temperature=temperature,
            left_boundary=left_boundary
        )

        self.regions.append(region)
        self.total_length += length



    def finalize(self):
        """
        Finalize the geometry and generate the mesh.

        This method:
        1. Creates all mesh cells for each region
        2. Places nodes at cell boundaries (including material interfaces)
        3. Assigns materials and temperatures to cells
        4. Cannot be undone - geometry becomes immutable
        """
        if self._finalized:
            print("Geometry already finalized")
            return

        if not self.regions:
            raise RuntimeError("No regions added to geometry")

        # Generate mesh cells for each region
        all_nodes = [0.0]  # Start with x=0
        cell_index = 0

        for region in self.regions:


            # Create cells for this region
            for i in range(region.n_cells):
                left_node = region.left_boundary + i * region.cell_width
                right_node = region.left_boundary + (i + 1) * region.cell_width
                center = (left_node + right_node) / 2.0

                cell = MeshCell(
                    index=cell_index,
                    region_index=region.index,
                    left_node=left_node,
                    right_node=right_node,
                    center=center,
                    width=region.cell_width,
                    material=region.material,
                    temperature=region.temperature
                )

                self.cells.append(cell)

                # Add right node (left node already added for first cell)
                all_nodes.append(right_node)

                cell_index += 1

        # Convert nodes to array and remove duplicates (material interfaces)
        self.nodes = np.array(sorted(set(all_nodes)))
        self.n_nodes = len(self.nodes)
        self.n_cells_total = len(self.cells)

        self._finalized = True


    def get_cell(self, index: int) -> MeshCell:
        """Get mesh cell by index."""
        if not self._finalized:
            raise RuntimeError("Geometry must be finalized first")

        if not (0 <= index < self.n_cells_total):
            raise IndexError(f"Cell index {index} out of range [0, {self.n_cells_total})")

        return self.cells[index]

    def get_region(self, index: int) -> MaterialRegion:
        """Get material region by index."""
        if not (0 <= index < len(self.regions)):
            raise IndexError(f"Region index {index} out of range [0, {len(self.regions)})")

        return self.regions[index]

    def find_cell_at_position(self, x: float) -> Optional[MeshCell]:
        """
        Find the mesh cell containing position x.

        Args:
            x: Position in cm

        Returns:
            MeshCell object or None if position is outside geometry
        """
        if not self._finalized:
            raise RuntimeError("Geometry must be finalized first")

        if x < 0 or x > self.total_length:
            return None

        # Binary search for cell
        for cell in self.cells:
            if cell.left_node <= x <= cell.right_node:
                return cell

        return None

    def get_macroscopic_xs_array(self, reaction: str = 'total') -> Dict[str, np.ndarray]:
        """
        Get macroscopic cross sections for all cells as arrays.

        Args:
            reaction: Reaction type

        Returns:
            Dictionary with:
                'xs': 2D array [n_cells, n_groups] of cross sections
                'positions': Array of cell center positions
                'cell_indices': Array of cell indices

        Example:
            >>> xs_data = geom.get_macroscopic_xs_array('total')
            >>> sigma_total = xs_data['xs']  # Shape: [n_cells, n_groups]
        """
        if not self._finalized:
            raise RuntimeError("Geometry must be finalized first")

        # Get number of energy groups from first cell
        first_xs = self.cells[0].material.get_macroscopic_xs(reaction)
        n_groups = len(first_xs)

        # Allocate arrays
        xs_array = np.zeros((self.n_cells_total, n_groups))
        positions = np.zeros(self.n_cells_total)
        cell_indices = np.arange(self.n_cells_total)

        # Fill arrays
        for i, cell in enumerate(self.cells):
            xs_array[i, :] = cell.material.get_macroscopic_xs(reaction)
            positions[i] = cell.center

        return {
            'xs': xs_array,
            'positions': positions,
            'cell_indices': cell_indices,
            'n_groups': n_groups,
            'n_cells': self.n_cells_total
        }

    def plot_geometry(self, figsize: Tuple[float, float] = (14, 4)):
        """
        Plot the geometry showing material regions and mesh.

        Args:
            figsize: Figure size

        Returns:
            matplotlib figure and axis
        """
        if not self._finalized:
            raise RuntimeError("Geometry must be finalized first")

        fig, ax = plt.subplots(figsize=figsize)

        # Color map for regions
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.regions)))

        # Plot each region
        for i, region in enumerate(self.regions):
            rect = plt.Rectangle(
                (region.left_boundary, 0),
                region.length,
                1.0,
                facecolor=colors[i],
                edgecolor='black',
                linewidth=2,
                label=f"{region.material.name} ({region.temperature:.0f} K)"
            )
            ax.add_patch(rect)

            # Add region label
            center_x = (region.left_boundary + region.right_boundary) / 2
            ax.text(center_x, 0.5, f"Region {i}\n{region.n_cells} cells",
                   ha='center', va='center', fontsize=10, fontweight='bold')

        # Plot mesh lines
        for node in self.nodes:
            ax.axvline(x=node, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

        # Formatting
        ax.set_xlim(-0.1, self.total_length + 0.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel('Position (cm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('', fontsize=12)
        ax.set_title(f'1D Geometry: {self.name}', fontsize=14, fontweight='bold')
        ax.set_yticks([])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(self.regions))
        ax.grid(True, axis='x', alpha=0.3)

        plt.tight_layout()
        return fig, ax

    def plot_temperature(self, figsize: Tuple[float, float] = (12, 5)):
        """
        Plot temperature distribution across geometry.

        Args:
            figsize: Figure size

        Returns:
            matplotlib figure and axis
        """
        if not self._finalized:
            raise RuntimeError("Geometry must be finalized first")

        fig, ax = plt.subplots(figsize=figsize)

        # Create step function for temperature
        positions = [cell.center for cell in self.cells]
        temperatures = [cell.temperature for cell in self.cells]

        ax.plot(positions, temperatures, 'o-', linewidth=2, markersize=6, color='red')

        # Add vertical lines at material interfaces
        for i in range(1, len(self.regions)):
            interface_x = self.regions[i].left_boundary
            ax.axvline(x=interface_x, color='black', linestyle='--', 
                      linewidth=1.5, alpha=0.7, label='Interface' if i == 1 else '')

        # Formatting
        ax.set_xlabel('Position (cm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Temperature (K)', fontsize=12, fontweight='bold')
        ax.set_title(f'Temperature Distribution: {self.name}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        return fig, ax

    def plot_cross_sections(self, reaction: str = 'total', 
                           energy_group: int = -1,
                           figsize: Tuple[float, float] = (12, 5)):
        """
        Plot cross section distribution across geometry.

        Args:
            reaction: Reaction type to plot
            energy_group: Energy group index (-1 for thermal, 0 for fast)
            figsize: Figure size

        Returns:
            matplotlib figure and axis
        """
        if not self._finalized:
            raise RuntimeError("Geometry must be finalized first")

        fig, ax = plt.subplots(figsize=figsize)

        # Get cross section data
        xs_data = self.get_macroscopic_xs_array(reaction)
        positions = xs_data['positions']
        xs_values = xs_data['xs'][:, energy_group]

        # Plot
        ax.plot(positions, xs_values, 'o-', linewidth=2, markersize=6)

        # Add vertical lines at material interfaces
        for i in range(1, len(self.regions)):
            interface_x = self.regions[i].left_boundary
            ax.axvline(x=interface_x, color='red', linestyle='--', 
                      linewidth=1.5, alpha=0.7)

        # Formatting
        group_label = f"Group {energy_group}" if energy_group >= 0 else "Thermal"
        ax.set_xlabel('Position (cm)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Σ_{{{reaction}}} (cm⁻¹)', fontsize=12, fontweight='bold')
        ax.set_title(f'{reaction.capitalize()} Cross Section ({group_label}): {self.name}',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        plt.tight_layout()
        return fig, ax

    def plot_all_groups(self, reaction: str = 'total',
                       figsize: Tuple[float, float] = (14, 8)):
        """
        Plot cross sections for all energy groups in 2D heatmap.

        Args:
            reaction: Reaction type
            figsize: Figure size

        Returns:
            matplotlib figure and axis
        """
        if not self._finalized:
            raise RuntimeError("Geometry must be finalized first")

        # Get cross section data
        xs_data = self.get_macroscopic_xs_array(reaction)
        xs_array = xs_data['xs']  # Shape: [n_cells, n_groups]
        positions = xs_data['positions']
        n_groups = xs_data['n_groups']

        fig, ax = plt.subplots(figsize=figsize)

        # Create 2D plot
        X, Y = np.meshgrid(positions, np.arange(1, n_groups + 1))

        # Take log of cross sections for better visualization
        xs_log = np.log10(xs_array.T + 1e-20)  # Transpose to [n_groups, n_cells]

        im = ax.pcolormesh(X, Y, xs_log, cmap='viridis', shading='auto')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label='log₁₀(Σ) [cm⁻¹]')

        # Add vertical lines at material interfaces
        for i in range(1, len(self.regions)):
            interface_x = self.regions[i].left_boundary
            ax.axvline(x=interface_x, color='red', linestyle='--', 
                      linewidth=2, alpha=0.7)

        # Formatting
        ax.set_xlabel('Position (cm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Energy Group', fontsize=12, fontweight='bold')
        ax.set_title(f'{reaction.capitalize()} Cross Sections (All Groups): {self.name}',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()
        return fig, ax

    def summary(self):
        """Print detailed geometry summary."""
        print("\n" + "="*75)
        print(f"1D Geometry Summary: {self.name}")
        print("="*75)

        if not self._finalized:
            print("Status: NOT FINALIZED")
            print(f"Regions added: {len(self.regions)}")
            print(f"Total length: {self.total_length:.4f} cm")
            print("\nCall finalize() to generate mesh")
        else:
            print("Status: FINALIZED")
            print(f"Total length: {self.total_length:.4f} cm")
            print(f"Total cells: {self.n_cells_total}")
            print(f"Total nodes: {self.n_nodes}")
            print(f"Number of regions: {len(self.regions)}")

            print("\nRegions:")
            print("-"*75)
            for region in self.regions:
                print(f"  Region {region.index}: {region.material.name}")
                print(f"    Position: [{region.left_boundary:.4f}, {region.right_boundary:.4f}] cm")
                print(f"    Length: {region.length:.4f} cm")
                print(f"    Cells: {region.n_cells}")
                print(f"    Cell width: {region.cell_width:.6f} cm")
                print(f"    Temperature: {region.temperature:.1f} K")
                print(f"    Nuclides: {len(region.material.components)}")
                print()

            print("Mesh Details:")
            print("-"*75)
            print(f"  Cell width range: [{min(c.width for c in self.cells):.6f}, "
                  f"{max(c.width for c in self.cells):.6f}] cm")
            print(f"  Average cell width: {np.mean([c.width for c in self.cells]):.6f} cm")

        print("="*75 + "\n")

    def __repr__(self):
        return (f"OneDimensionalCartesianGeometryAndMesh(name='{self.name}', "
                f"regions={len(self.regions)}, finalized={self._finalized})")

    def __len__(self):
        """Return total number of cells."""
        return self.n_cells_total


# In[5]:


"""Cross-Section Post-Processor
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
            for g_from in range(self.n_groups):
                row = f"{indent}  From G{g_from+1:<3}"
                for g_to in range(self.n_groups):
                    row += f"{self.scatter_matrix[g_from, g_to]:<12.6e}"
                print(row)

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

        # Extract fine-group cross sections per cell
        self._extract_fine_group_xs()

        # Standard fission spectrum (8-group MPACT)
        self.chi_fine = np.array([0.584349, 0.415378, 0.000272, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Average cosine for transport XS (P1 scattering approx)
        # For hydrogen-dominated systems, μ̄ ≈ 2/(3A) where A is mass number
        # We'll compute this from the scattering data if available, or use 0 (isotropic)
        self.mu_bar = np.zeros((self.n_cells, self.n_groups))


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

        if total_phi_V < 1e-30:
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



        # Get region information
        regions = self.geom.regions

        # 1-group by region
        for region_idx, region in enumerate(regions):
            cell_indices = [c.index for c in self.geom.cells if c.region_index == region_idx]
            if cell_indices:
                name = f"1-Group: {region.material.name}"
                results[f'1g_region_{region_idx}'] = self.collapse_to_1group(
                    cell_indices=cell_indices, name=name
                )
                # results[f'1g_region_{region_idx}'].print_summary("  ")

        # 1-group whole problem
        results['1g_total'] = self.collapse_to_1group(name="1-Group: Homogenized Total")
        # results['1g_total'].print_summary("  ")

        # 2-group by region
        for region_idx, region in enumerate(regions):
            cell_indices = [c.index for c in self.geom.cells if c.region_index == region_idx]
            if cell_indices:
                name = f"2-Group: {region.material.name}"
                results[f'2g_region_{region_idx}'] = self.collapse_to_2group(
                    cell_indices=cell_indices, name=name
                )
                # results[f'2g_region_{region_idx}'].print_summary("  ")

        # 2-group whole problem
        results['2g_total'] = self.collapse_to_2group(name="2-Group: Homogenized Total")
        # results['2g_total'].print_summary("  ")

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


    # Load library
    lib = MPACTLibrary('mpact8g_70s_v4_0m0_02232015.fmt')

    # Create a simple 2-region problem: UO2 + H2O

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
    # print("\nSolving transport problem...")
    solver = LDFESNTransportSolver(geom, quadrature_order=8,
                                   left_bc='reflecting', right_bc='reflecting')
    solver.max_iterations = 5000
    solver.solve()
    # solver.print_summary()


    pp = CrossSectionPostProcessor(solver)
    results = pp.process_all()

    return results


# In[6]:


"""Pincell Generator
"""
def create_pwr_pin_cell(lib: MPACTLibrary,poison_fraction = 1E-10,enrichment_multiplier=1):
    """
    Create a standard PWR pin cell geometry.

    Typical 17x17 Westinghouse PWR parameters:
    - Fuel pellet radius: 0.4096 cm
    - Gap thickness: 0.0082 cm  
    - Clad inner radius: 0.4178 cm
    - Clad outer radius: 0.4750 cm
    - Pin pitch: 1.26 cm

    Returns:
        geometry, dict of materials
    """


    # Dimensions (cm)
    fuel_radius = 0.4096
    gap_thickness = 0.0082
    clad_thickness = 0.0572  # 0.4750 - 0.4178
    pitch = 1.26
    half_pitch = pitch / 2.0

    # Moderator thickness (from clad outer to half-pitch)
    clad_outer_radius = fuel_radius + gap_thickness + clad_thickness
    mod_thickness = half_pitch - clad_outer_radius

    print(f"\nPin Cell Dimensions:")
    print(f"  Fuel radius:     {fuel_radius:.4f} cm")
    print(f"  Gap thickness:   {gap_thickness:.4f} cm")
    print(f"  Clad thickness:  {clad_thickness:.4f} cm")
    print(f"  Clad outer:      {clad_outer_radius:.4f} cm")
    print(f"  Half-pitch:      {half_pitch:.4f} cm")
    print(f"  Moderator:       {mod_thickness:.4f} cm")

    # Get nuclides
    u235 = lib.find_nuclide_by_name('U-235')
    u238 = lib.find_nuclide_by_name('U-238')
    o16 = lib.find_nuclide_by_name('O-16')
    h1 = lib.find_nuclide_by_name('H-1')
    b10 = lib.find_nuclide_by_name('B-10')
    b11 = lib.find_nuclide_by_name('B-11')
    zr_nat = lib.find_nuclide_by_name('ZR-NAT')
    he4 = lib.find_nuclide_by_name('HE-4')

    # =========================================================================
    # UO2 Fuel (5% enriched, 95% TD)
    # =========================================================================
    fuel_temp = 900.0  # K (hot fuel)
    fuel_density = 10.4  # g/cm³ (95% of 10.97 theoretical density)

    # Number density calculation for UO2
    # N_UO2 = ρ * N_A / M_UO2 where M_UO2 ≈ 270 g/mol
    # At 10.97 g/cm³: N_UO2 ≈ 0.0245 molecules/b-cm
    # At 10.4 g/cm³ (95% TD): N_UO2 ≈ 0.0232 molecules/b-cm
    N_uo2 = 0.0232  # atoms/b-cm (total UO2 molecules)
    enrichment = 0.05*enrichment_multiplier  # 5% U-235

    uo2 = Material("UO2 Fuel (5% enr)", temperature=fuel_temp, density=fuel_density)
    uo2.add_nuclide(u235, N_uo2 * enrichment / 3.0)      # 1 U per UO2, 5% is U-235
    uo2.add_nuclide(u238, N_uo2 * (1-enrichment) / 3.0)  # 95% is U-238
    uo2.add_nuclide(o16, N_uo2 * 2.0 / 3.0)              # 2 O per UO2


    # =========================================================================
    # Helium Gap
    # =========================================================================
    gap_temp = 600.0  # K

    # He at ~2.5 MPa, 600K: very low density
    # N_He ≈ P/(kT) in atoms/b-cm, approximately 1e-4 to 1e-3
    N_he = 1.0e-4  # atoms/b-cm (low density helium)

    he_gap = Material("He Gap", temperature=gap_temp, density=0.001)
    if he4:
        he_gap.add_nuclide(he4, N_he)
    else:
        # Fallback: use very dilute H if He not available
        he_gap.add_nuclide(h1, 1.0e-6)
        print("  Warning: HE-4 not found, using trace H-1")


    # =========================================================================
    # Zircaloy-4 Cladding
    # =========================================================================
    clad_temp = 600.0  # K
    clad_density = 6.55  # g/cm³

    # Zircaloy-4: ~98.2% Zr, 1.5% Sn, 0.2% Fe, 0.1% Cr
    # N_Zr = ρ * N_A / M_Zr where M_Zr ≈ 91.22 g/mol
    # At 6.55 g/cm³: N_Zr ≈ 0.0432 atoms/b-cm
    N_zr_total = 0.0432

    zircaloy = Material("Zircaloy-4", temperature=clad_temp, density=clad_density)
    if zr_nat:
        zircaloy.add_nuclide(zr_nat, N_zr_total * 0.982)

    # Add Sn and Fe if available
    sn_nat = lib.find_nuclide_by_name('SN-NAT')
    fe56 = lib.find_nuclide_by_name('FE-56')
    if sn_nat:
        zircaloy.add_nuclide(sn_nat, N_zr_total * 0.015)
    if fe56:
        zircaloy.add_nuclide(fe56, N_zr_total * 0.003)


    # =========================================================================
    # H2O Moderator (PWR conditions)
    # =========================================================================
    mod_temp = 580.0  # K (~307°C, typical PWR inlet)
    mod_density = 0.71  # g/cm³ (PWR operating conditions, ~15.5 MPa)

    # H2O number density
    # N_H2O = ρ * N_A / M_H2O where M_H2O = 18 g/mol
    # At 1.0 g/cm³: N_H2O ≈ 0.0334 molecules/b-cm → N_total ≈ 0.1003 atoms/b-cm
    # At 0.71 g/cm³: scale by density
    N_h2o_ref = 0.1003  # atoms/b-cm at 1.0 g/cm³
    N_h2o = N_h2o_ref * mod_density

    h2o = Material("H2O Moderator", temperature=mod_temp, density=mod_density)
    h2o.add_nuclide(b10, N_h2o*poison_fraction)
    h2o.add_nuclide(h1,  N_h2o * (2.0 / 3.0)*(1-poison_fraction))
    h2o.add_nuclide(o16, N_h2o * (1.0 / 3.0)*(1-poison_fraction))


    # =========================================================================
    # Create Geometry (1D slab approximation of pin cell)
    # =========================================================================
    # Using reflecting BC on both sides represents infinite lattice

    geom = OneDimensionalCartesianGeometryAndMesh("PWR Pin Cell")

    # Mesh: finer in fuel, coarser elsewhere
    geom.add_region(uo2, length=fuel_radius, n_cells=20, temperature=fuel_temp)
    geom.add_region(he_gap, length=gap_thickness, n_cells=2, temperature=gap_temp)
    geom.add_region(zircaloy, length=clad_thickness, n_cells=4, temperature=clad_temp)
    geom.add_region(h2o, length=mod_thickness, n_cells=15, temperature=mod_temp)

    geom.finalize()

    materials = {
        'fuel': uo2,
        'gap': he_gap,
        'clad': zircaloy,
        'moderator': h2o
    }

    return geom, materials


# # Student Input Here: Question 3
# 
# Below is the code that will run a multigroup simulation of a pincell problem and generate one-group cross-sections. It then uses those cross-sections to estimate k $_{\infty}$.<br>
# The pincell already has default values for all parameters necessary, but allows the user to input two factors: fuel enrichment multiplier and boron-10 fraction in the water (boron-10 is a neutron absorber). <br>
# Make sure to read the code comments. <br><br>
# What we want is **plots of your chosen input values** against the value of the variable **k_inf_approx**, which is already defined for you.
# 
# 3a) Run the simulation for several (at least 4) different fuel enrichment multipliers and make a plot of the resulting k $_{\infty}$ vs the fuel enrichment. 
# 
# 3b) Run the simulation for several (at least 4) different boron-10 fractions and make a plot of the resulting k $_{\infty}$ vs the boron-10 fraction. 
# 
# 3c) Explain the behavior exhibited in the graphs. Why does the criticality of the system behave in this way?

# <br>
# 
# *Response to 3c here (double-click to edit)*
# <br><br><br>
# 
# More enrichment increases the amount of fissile material in the reactor meaning neutrons are more likly to hit fissile material rather than abosorber material becuase the fissile material macroscopic crosssection is larger. This means every neutron created is more likly to create more neutrons so on average more neutrons are created in the next generation increasing k_inf.
# 
# More boron increases the amount of absorber material in the reactor meaning neutrons are less likly to hit fissile material material becuase the absorber material macroscopic crosssection is larger. This means every neutron created is less likly to create more neutrons so on average less neutrons are created in the next generation decreasing k_inf.

# In[7]:


def run_pwr_pin_cell_simulation(enrichment_multiplier=1,poison_fraction=1E-10):
    """Run a PWR pin cell simulation and compute k-infinity approximation."""
    #from mpact_reader import MPACTLibrary
    #from mpact_transport_solver import LDFESNTransportSolver


    # Load MPACT library
    mpact_path = 'mpact_data_8g.fmt'
    lib = MPACTLibrary(mpact_path)


    #done this way to allow more even sampling across the range

    # Create geometry
    #poison_fraction = 1E-10 #default value
    #enrichment_mult = 1 #default value since it's a multiplier. the pincell has an enrichment of 0.05 by default

    #this is the function that will create the pincell
    geom, materials = create_pwr_pin_cell(lib,poison_fraction=poison_fraction,enrichment_multiplier=enrichment_mult)

    # =========================================================================
    # Solve Transport Problem: Do not edit this part
    # =========================================================================

    solver = LDFESNTransportSolver(
        geom, 
        quadrature_order=4,      
        left_bc='reflecting', 
        right_bc='reflecting'
    )
    solver.tolerance = 1e-5      
    solver.max_iterations = 5000

    solver.solve()

    pp = CrossSectionPostProcessor(solver)
    results = pp.process_all()

    xs_total = results['2g_total']

    nu_sigma_f_total = xs_total.nu_sigma_f.sum()
    sigma_a_total = xs_total.sigma_a.sum()

    # For thermal reactor: k_inf ≈ η * f * p * ε
    # Or simply: k_inf ≈ (νΣ_f1 + νΣ_f2) / (Σ_a1 + Σ_a2) for homogenized

    # =========================================================================
    # End of Transport Problem Solve
    # =========================================================================

    #this is your output that you must save and plot in the next block 
    k_inf_approx = nu_sigma_f_total / sigma_a_total
    return k_inf_approx


""" # In[8]:


#use some sort of plotting in here, likely matplotlib
enrichment_values = [0.4, 0.5, 0.6, 0.7,0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]  # Multipliers for 5% enrichment
boron_values = [1E-12, 1E-11, 1E-10, 1E-9, 1E-8, 1E-7, 1E-6, 1E-5]  # Boron-10 fractions in moderator
enrichment_results = []
boron_results = []
# Run simulations for varying boron-10 fractions
for enrichment_mult in enrichment_values:
    k_inf = run_pwr_pin_cell_simulation(enrichment_multiplier=enrichment_mult,poison_fraction=1E-10)
    enrichment_results.append(k_inf)  

for boron_frac in boron_values:
    k_inf = run_pwr_pin_cell_simulation(enrichment_multiplier=1,poison_fraction=boron_frac)
    boron_results.append(k_inf)



# In[ ]:


# 3a) Plot k_inf vs fuel enrichment (actual enrichment = 0.05 * enrichment_multiplier)
enrichment_plot_x = [0.05 * mult for mult in enrichment_values]
print(enrichment_plot_x)



# In[11]:


enrichment_plot_y = [enrichment_results[i] for i in range(len(enrichment_plot_x))]

boron_plot_x = boron_values
boron_plot_y = [boron_results[i] for i in range(len(boron_plot_x))]


# In[13]:


plt.figure(figsize=(7,5))
plt.plot(enrichment_plot_x, enrichment_plot_y, marker='o', linestyle='-')
plt.xlabel('U-235 Enrichment (fraction)')
plt.ylabel(r'$k_{\infty}$ approximation')
plt.title(r'$k_{\infty}$ vs U-235 Enrichment')
plt.grid(True)
plt.show()

# 3b) Plot k_inf vs boron-10 fraction

plt.figure(figsize=(7,5))
plt.plot(boron_plot_x, boron_plot_y, marker='s', linestyle='-')
plt.xlabel('Boron-10 Fraction in Moderator')
plt.ylabel(r'$k_{\infty}$ approximation')
plt.title(r'$k_{\infty}$ vs Boron-10 Fraction')
plt.xscale('log')
plt.grid(True)
plt.show()



#  """
