"""
Simple 1D multigroup diffusion solver for the ME388D final project.

This module reuses the existing geometry/material infrastructure so the same
pin-cell setup used by the Sn solver can also be solved with diffusion theory.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from mpact_geometry import OneDimensionalCartesianGeometryAndMesh
from mpact_reader import MPACTLibrary
from sn_transport import create_pwr_pin_cell


@dataclass
class DiffusionSolution:
    """Container for the diffusion eigenvalue solution."""

    k_eff: float
    flux: np.ndarray
    iterations: int
    converged: bool
    cell_centers: np.ndarray


class MultigroupDiffusionSolver:
    """Cell-centered finite-difference multigroup diffusion solver."""

    def __init__(
        self,
        geom: OneDimensionalCartesianGeometryAndMesh,
        left_bc: str = "reflecting",
        right_bc: str = "reflecting",
    ):
        if not geom._finalized:
            raise RuntimeError("Geometry must be finalized before solving.")

        self.geom = geom
        self.left_bc = left_bc.lower()
        self.right_bc = right_bc.lower()
        self.n_cells = geom.n_cells_total
        self.n_groups = geom.cells[0].material.get_macroscopic_xs("total").size
        self.widths = np.array([cell.width for cell in geom.cells], dtype=float)
        self.centers = np.array([cell.center for cell in geom.cells], dtype=float)

        self.diffusion = np.zeros((self.n_cells, self.n_groups))
        self.sigma_t = np.zeros((self.n_cells, self.n_groups))
        self.sigma_a = np.zeros((self.n_cells, self.n_groups))
        self.nu_sigma_f = np.zeros((self.n_cells, self.n_groups))
        self.chi = np.zeros((self.n_cells, self.n_groups))
        self.scatter = np.zeros((self.n_cells, self.n_groups, self.n_groups))

        self._extract_cross_sections()
        self.loss_matrix = self._build_loss_matrix()
        self.fission_matrix = self._build_fission_matrix()

    def _extract_cross_sections(self):
        """Build cell-wise multigroup cross sections from the existing materials."""
        default_chi = np.array(
            [0.5843493, 0.4153784, 0.0002723212, 0.0, 0.0, 0.0, 0.0, 0.0]
        )

        for i, cell in enumerate(self.geom.cells):
            material = cell.material
            self.sigma_t[i, :] = material.get_macroscopic_xs("total")
            self.sigma_a[i, :] = material.get_macroscopic_xs("absorption")

            try:
                sigma_tr = material.get_macroscopic_xs("transport")
            except Exception:
                sigma_tr = self.sigma_t[i, :].copy()

            sigma_tr = np.clip(sigma_tr, 1.0e-8, None)
            self.diffusion[i, :] = 1.0 / (3.0 * sigma_tr)

            try:
                self.nu_sigma_f[i, :] = material.get_macroscopic_xs("nu-fission")
            except Exception:
                sigma_f = material.get_macroscopic_xs("fission")
                nu_guess = np.linspace(2.8, 2.4, self.n_groups)
                self.nu_sigma_f[i, :] = nu_guess * sigma_f

            for component in material.components.values():
                nuclide = component.nuclide
                if nuclide.scatter_matrix.shape == (self.n_groups, self.n_groups):
                    self.scatter[i, :, :] += component.number_density * nuclide.scatter_matrix

            self.chi[i, :] = default_chi / default_chi.sum()

    def _interface_diffusion(self, left_value: float, right_value: float) -> float:
        """Harmonic average at a material interface."""
        if left_value <= 0.0 or right_value <= 0.0:
            return 0.0
        return 2.0 * left_value * right_value / (left_value + right_value)

    def _build_loss_matrix(self) -> np.ndarray:
        """Assemble the diffusion loss operator."""
        size = self.n_cells * self.n_groups
        matrix = np.zeros((size, size))

        for g in range(self.n_groups):
            for i in range(self.n_cells):
                row = g * self.n_cells + i
                dx = self.widths[i]

                sigma_removal = self.sigma_t[i, g] - self.scatter[i, g, g]
                matrix[row, row] += sigma_removal

                if i > 0:
                    d_face = self._interface_diffusion(
                        self.diffusion[i - 1, g], self.diffusion[i, g]
                    )
                    dx_face = 0.5 * (self.widths[i - 1] + self.widths[i])
                    coeff = d_face / (dx * dx_face)
                    matrix[row, row] += coeff
                    matrix[row, row - 1] -= coeff
                elif self.left_bc == "vacuum":
                    matrix[row, row] += 2.0 * self.diffusion[i, g] / (dx * dx)

                if i < self.n_cells - 1:
                    d_face = self._interface_diffusion(
                        self.diffusion[i, g], self.diffusion[i + 1, g]
                    )
                    dx_face = 0.5 * (self.widths[i] + self.widths[i + 1])
                    coeff = d_face / (dx * dx_face)
                    matrix[row, row] += coeff
                    matrix[row, row + 1] -= coeff
                elif self.right_bc == "vacuum":
                    matrix[row, row] += 2.0 * self.diffusion[i, g] / (dx * dx)

                for gp in range(self.n_groups):
                    if gp == g:
                        continue
                    scatter_in = self.scatter[i, gp, g]
                    if scatter_in != 0.0:
                        col = gp * self.n_cells + i
                        matrix[row, col] -= scatter_in

        return matrix

    def _build_fission_matrix(self) -> np.ndarray:
        """Assemble the production operator."""
        size = self.n_cells * self.n_groups
        matrix = np.zeros((size, size))

        for g in range(self.n_groups):
            for gp in range(self.n_groups):
                block = np.diag(self.chi[:, g] * self.nu_sigma_f[:, gp])
                row_slice = slice(g * self.n_cells, (g + 1) * self.n_cells)
                col_slice = slice(gp * self.n_cells, (gp + 1) * self.n_cells)
                matrix[row_slice, col_slice] = block

        return matrix

    def solve(
        self,
        max_iterations: int = 200,
        tolerance: float = 1.0e-6,
        initial_flux: Optional[np.ndarray] = None,
    ) -> DiffusionSolution:
        """Solve the multigroup k-eigenvalue problem by power iteration."""
        size = self.n_cells * self.n_groups
        if initial_flux is None:
            flux = np.ones(size)
        else:
            flux = np.asarray(initial_flux, dtype=float).reshape(size)

        flux /= np.linalg.norm(flux)
        k_eff = 1.0
        converged = False

        for iteration in range(1, max_iterations + 1):
            source = self.fission_matrix @ flux
            updated_flux = np.linalg.solve(self.loss_matrix, source / k_eff)

            production_old = float(np.sum(source))
            production_new = float(np.sum(self.fission_matrix @ updated_flux))
            if production_old <= 0.0 or production_new <= 0.0:
                raise RuntimeError("Non-positive fission source encountered.")

            k_new = k_eff * production_new / production_old
            updated_flux /= np.linalg.norm(updated_flux)

            flux_error = np.linalg.norm(updated_flux - flux) / max(
                np.linalg.norm(updated_flux), 1.0e-12
            )
            k_error = abs(k_new - k_eff) / max(abs(k_new), 1.0e-12)

            flux = updated_flux
            k_eff = k_new

            if max(flux_error, k_error) < tolerance:
                converged = True
                break

        flux = flux.reshape(self.n_groups, self.n_cells).T
        return DiffusionSolution(
            k_eff=k_eff,
            flux=flux,
            iterations=iteration,
            converged=converged,
            cell_centers=self.centers.copy(),
        )


def run_pwr_pin_cell_diffusion(
    mpact_path: str = "mpact_data_8g.fmt",
    enrichment_multiplier: float = 1.0,
    poison_fraction: float = 1.0e-10,
    left_bc: str = "reflecting",
    right_bc: str = "reflecting",
) -> DiffusionSolution:
    """Convenience wrapper matching the existing pin-cell setup."""
    lib = MPACTLibrary(mpact_path)
    geom, _ = create_pwr_pin_cell(
        lib,
        poison_fraction=poison_fraction,
        enrichment_multiplier=enrichment_multiplier,
    )
    solver = MultigroupDiffusionSolver(geom, left_bc=left_bc, right_bc=right_bc)
    return solver.solve()


def summarize_flux(solution: DiffusionSolution) -> Dict[str, np.ndarray]:
    """Return a few useful derived quantities for quick plotting or inspection."""
    flux = solution.flux
    total_flux = flux.sum(axis=1)
    normalized_total_flux = total_flux / max(total_flux.max(), 1.0e-12)
    return {
        "cell_centers": solution.cell_centers,
        "group_flux": flux,
        "total_flux": total_flux,
        "normalized_total_flux": normalized_total_flux,
    }


if __name__ == "__main__":
    solution = run_pwr_pin_cell_diffusion()
    print(f"k_eff = {solution.k_eff:.6f}")
    print(f"iterations = {solution.iterations}")
    print(f"converged = {solution.converged}")
