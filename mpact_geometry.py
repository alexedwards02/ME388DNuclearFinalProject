"""
One-Dimensional Cartesian Geometry and Mesh
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
from mpact_reader import MPACTLibrary
from mpact_material import Material


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
        
        print(f"Added region {region.index}: {material.name}, "
              f"x=[{left_boundary:.4f}, {region.right_boundary:.4f}] cm, "
              f"{n_cells} cells, T={temperature:.1f} K")
    
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
        
        print(f"\nFinalizing geometry: {self.name}")
        print("="*70)
        
        # Generate mesh cells for each region
        all_nodes = [0.0]  # Start with x=0
        cell_index = 0
        
        for region in self.regions:
            print(f"\nRegion {region.index}: {region.material.name}")
            print(f"  Position: [{region.left_boundary:.4f}, {region.right_boundary:.4f}] cm")
            print(f"  Cells: {region.n_cells}, Width: {region.cell_width:.6f} cm")
            
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
        
        print(f"\n" + "="*70)
        print(f"Geometry finalized:")
        print(f"  Total length: {self.total_length:.4f} cm")
        print(f"  Total cells: {self.n_cells_total}")
        print(f"  Total nodes: {self.n_nodes}")
        print(f"  Regions: {len(self.regions)}")
        print("="*70)
    
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


if __name__ == "__main__":
    print("OneDimensionalCartesianGeometryAndMesh")
    print("="*50)
    print("\nExample usage:")
    print("  from mpact_geometry import OneDimensionalCartesianGeometryAndMesh")
    print("  geom = OneDimensionalCartesianGeometryAndMesh('Reactor Core')")
    print("  geom.add_region(fuel, length=2.0, n_cells=20, temperature=900)")
    print("  geom.add_region(moderator, length=1.0, n_cells=10, temperature=350)")
    print("  geom.finalize()")
    print("  geom.plot_geometry()")
