""" Define a way of visualising various items (plotter, essentially) """
from __future__ import annotations

from logging import Logger

import astropy.constants as const
import astropy.units as u
import numpy as np
import rebound
from matplotlib import pylab as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

# if TYPE_CHECKING:
from planetary_system.system import System


class SystemVisualiser:
    def __init__(self, _system_: System = None) -> None:
        if _system_:
            self.system = _system_

    @property
    def system(self) -> System:
        if hasattr(self, "_system_"):
            return self._system_

    @system.setter
    def system(self, _system_: System) -> None:
        self._system_ = _system_
        if _system_ and _system_.plotter is not self:
            _system_.plotter = self

    @property
    def logger(self) -> Logger:
        if self.system:
            return self.system.logger

    def plot_system(self, style: str = "complex") -> None:
        if not hasattr(self.system, "simulation"):
            return self.plot_simple()
        match style.lower():
            case "simple":
                return self.plot_simple()
            case "complex":
                return self.plot_complex()
        raise Exception(f"Unknown plotting style {style}")

    def plot_simple(self) -> None:
        max_extent = max(body.orbit.a for body in self.system.flattened if body.orbit)
        scale = 1 * u.m if max_extent <= 0.01 * const.au else const.au

        fig, ax = plt.subplots(1, 1)
        for body in self.system.flattened:
            x, y, z = body.pos
            colour = getattr(body, "normalised_colour", (0, 0, 0))

            if body.orbit:
                plotting_points = 360
                rx, ry = (body.orbit.orbit_trail(plotting_points) / scale).value
                orbit = np.zeros((plotting_points - 1, 2, 2))
                orbit[:, 0, 0] = rx[:-1]
                orbit[:, 0, 1] = ry[:-1]
                orbit[:, 1, 0] = rx[1:]
                orbit[:, 1, 1] = ry[1:]
                orbit_colour = np.zeros((plotting_points, 4))
                orbit_colour[:, 0:3] = colour
                orbit_colour[:, 3] = np.linspace(0, 1, plotting_points)[::-1]
                self.logger.debug(orbit)
                ax.add_collection(LineCollection(orbit, color=orbit_colour, lw=1))

            ax.add_artist(
                plt.Circle(
                    ((x / scale).value, (y / scale).value), radius=0.1, color=colour
                )
            )

        lim = (max_extent / scale).value * 1.2
        ax.set_xlim((-lim, lim))
        ax.set_ylim((-lim, lim))

        plt.title(self.system.name)

        plt.show()

    def _sim_parents_(self) -> dict[int, list[int]]:
        parents = {}
        bodies = self.system.flattened
        for idx, body in enumerate(bodies):
            if parent := getattr(body, "parent"):
                if parent not in bodies:
                    continue
                parent_idx = bodies.index(parent)
                if parent_idx in parents:
                    parents[parent_idx] += [idx]
                else:
                    parents[parent_idx] = [idx]
        return parents[0], {k: v for k, v in parents.items() if k != 0}

    def plot_complex(self) -> None:
        sim = self.system.simulation.sim
        main_sim, sub_sims = self._sim_parents_()
        lim = (
            max(body.orbit.a for body in self.system.flattened if body.orbit) / const.au
        ).value * 1.2
        op = rebound.OrbitPlotSet(
            sim,
            particles=main_sim,
            xlim=[-lim, lim],
            ylim=[-lim, lim],
            color=True,
            figsize=(10, 10),
            unitlabel="[AU]",
        )
        for parent, sub_sim in sub_sims.items():
            rebound.OrbitPlotSet(
                sim,
                particles=sub_sim,
                primary=parent,
                show_primary=False,
                fig=op.fig,
                ax=(op.ax_main, op.ax_top, op.ax_right),
                xlim=[-lim, lim],
                ylim=[-lim, lim],
                color=True,
                figsize=(10, 10),
                unitlabel="[AU]",
            )
        op.fig.suptitle(self.system.name.capitalize())

        colours = [
            (1.0, 0.0, 0.0),
            (0.0, 0.75, 0.75),
            (0.75, 0.0, 0.75),
            (
                0.75,
                0.75,
                0,
            ),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.5, 0.0),
        ]
        legend_elements = [
            Line2D(
                [0],
                [0],
                color=colours[idx % len(colours)],
                lw=1.5,
                label=f"{body.name} orbit",
            )
            for idx, body in enumerate(self.system.flattened[1:])
        ]

        plt.legend(
            handles=legend_elements,
            framealpha=0,
            bbox_to_anchor=(0, 1.5),
            loc="upper left",
        )
        op.fig.tight_layout()
        plt.show()
