"""
MPACT Formatted Library Reader
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
        
        print(f"  Chi spectrum: {self.chi_spectrum}")
    
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


if __name__ == "__main__":
    # Test the reader
    lib = MPACTLibrary('mpact8g_70s_v4.0m0_02232015.fmt')
    
    print("\n" + "="*75)
    print("Nuclide List (first 20)")
    print("="*75)
    nuclides = lib.list_nuclides()
    for nuc_id, nuc_name in nuclides[:20]:
        print(f"  ID {nuc_id:6d}: {nuc_name}")
    
    # Test U-235
    print("\n" + "="*75)
    print("U-235 Data")
    print("="*75)
    u235 = lib.find_nuclide_by_name('U-235')
    if u235:
        print(f"ID: {u235.id}")
        print(f"Atomic mass: {u235.atomic_mass:.2f} amu")
        print(f"Temperatures: {u235.temperatures}")
        print(f"\nCross sections at {u235.temperatures[0]:.0f} K:")
        for g in range(8):
            print(f"  Group {g+1}: σ_t={u235.cross_sections['total'][g]:.4f}, "
                  f"σ_a={u235.cross_sections['absorption'][g]:.4f}, "
                  f"νσ_f={u235.cross_sections['nu-fission'][g]:.4f}")
        
        print(f"\nScattering matrix (diagonal and down-scatter):")
        for g_from in range(8):
            scatter_row = []
            for g_to in range(8):
                if u235.scatter_matrix[g_from, g_to] > 1e-6:
                    scatter_row.append(f"{g_to+1}:{u235.scatter_matrix[g_from, g_to]:.4f}")
            if scatter_row:
                print(f"  Group {g_from+1} → {', '.join(scatter_row)}")
