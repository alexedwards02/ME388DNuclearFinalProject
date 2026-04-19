"""
Material Class for MPACT Library
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
from mpact_reader import MPACTLibrary, MPACTNuclide


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


if __name__ == "__main__":
    print("Material Class for MPACT Library")
    print("=" * 50)
    print("\nExample usage:")
    print("  from mpact_reader import MPACTLibrary")
    print("  from mpact_material import Material")
    print("  ")
    print("  lib = MPACTLibrary('mpact8g_70s_v4_0m0_02232015.fmt')")
    print("  mat = Material('UO2 Fuel')")
    print("  ")
    print("  u235 = lib.find_nuclide_by_name('U-235')")
    print("  mat.add_nuclide(u235, number_density=1.5e-3)")
    print("  ")
    print("  sigma_total = mat.get_macroscopic_xs('total')")
    print("  mat.plot_macroscopic_xs()")
