from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from typing import Any, Type

import astropy.constants as const
import astropy.units as u

from common import log_children

from .bodies import _bodycontainer_, empty_3_vector
from .bodies import presets as preset_bodies
from .orbits import presets as preset_orbits

logger = getLogger("system")


@dataclass
class _system_(_bodycontainer_):
    def __str__(self) -> str:
        return f"{super().__str__()} containing {len(self.flattened)} objects"

    def log_bodies(self) -> None:
        log_children(self.logger, self, "_bodies_")

    @property
    def plotter(self) -> Any:
        if hasattr(self, "_plotter_"):
            return self._plotter_

    @plotter.setter
    def plotter(self, plotter: Any) -> None:
        self._plotter_ = plotter
        if plotter and plotter.system is not self:
            plotter.system = self

    def plot_system(self, style: str = "complex") -> None:
        """plot the current position of the planets in the system"""
        self.plotter.plot_system(style=style)

    @property
    def simulation(self) -> Any:
        if hasattr(self, "_simulation_"):
            return self._simulation_

    @simulation.setter
    def simulation(self, simulation: Any) -> None:
        self._simulation_ = simulation
        if simulation and simulation.system is not self:
            simulation.system = self

    def simulate(self, time: u.Quantity[u.s]) -> None:
        self.simulation.simulate(time)

    def fetch_body(self, name: str) -> Type[_bodycontainer_]:
        return [body for body in self.flattened if body.name == name][0]

    @property
    def total_mass(self) -> u.Quantity[u.kg]:
        return sum(body.mass for body in self.flattened)

    @property
    def centre_of_mass(
        self,
    ) -> tuple[u.Quantity[u.m], u.Quantity[u.m], u.Quantity[u.m]]:
        pos = empty_3_vector * u.m * u.kg
        for body in self.flattened:
            pos += body.pos * body.mass
        return pos / self.total_mass

    def distance(
        self, primary: Type[_bodycontainer_], secondary: Type[_bodycontainer_]
    ) -> u.Quantity[u.m]:
        return (sum((primary.pos - secondary.pos) ** 2)) ** 0.5

    def gravitation(
        self, primary: Type[_bodycontainer_], secondary: Type[_bodycontainer_]
    ) -> u.Quantity[u.N]:
        if primary == self or secondary == self:
            return 0 * u.N
        return (
            const.G
            * primary.mass
            * secondary.mass
            / (self.distance(primary, secondary) + 1 * u.mm) ** 2
        )

    def find_parent(self, child: Type[_bodycontainer_]) -> Type[_bodycontainer_]:
        parent = self
        strength = 0 * u.N
        for body in self.flattened:
            grav = self.gravitation(body, child)
            if grav > strength:
                parent = body
                strength = grav
        return parent

    def compute_hierarchy(self) -> None:
        bodies = self.flattened
        for body, parent in [(body, self.find_parent(body)) for body in bodies]:
            if body.parent:
                body.parent.remove_body(body)
            parent.add_body(body)


@dataclass
class System(_system_):
    pass


class presets:
    def _earthMoon_():
        class barycentre(System):
            pass

        bary = barycentre("Earth-moon system")
        earth = preset_bodies.Earth
        moon = preset_bodies.Moon
        moon.orbit = preset_orbits.Moon
        bary.add_body(earth)
        earth.add_body(moon)
        return bary

    def _innerPlanets_():
        class innerBodies(System):
            pass

        sol = innerBodies("solar system Rocky planets")
        sun = preset_bodies.Sun
        mercury = preset_bodies.Mercury
        venus = preset_bodies.Venus
        earth = preset_bodies.Earth
        mars = preset_bodies.Mars
        mercury.orbit = preset_orbits.Mercury
        venus.orbit = preset_orbits.Venus
        earth.orbit = preset_orbits.Earth
        mars.orbit = preset_orbits.Mars
        sol.add_body(sun)
        sun.add_body(mercury)
        sun.add_body(venus)
        sun.add_body(earth)
        sun.add_body(mars)
        return sol

    # def _outerPlanets_():
    #     class outerBodies(System):
    #         pass
    #     sol = outerBodies("solar system Gas Giants")
    #     sun = preset_bodies.Sun
    #     jupiter = preset_bodies.Jupiter
    #     saturn = preset_bodies.Saturn
    #     uranus = preset_bodies.Uranus
    #     neptune = preset_bodies.Neptune
    #     jupiter.orbit = preset_orbits.Jupiter
    #     saturn.orbit = preset_orbits.Saturn
    #     uranus.orbit = preset_orbits.Uranus
    #     neptune.orbit = preset_orbits.Neptune
    #     sol.add_body(sun)
    #     sun.add_body(jupiter)
    #     sun.add_body(saturn)
    #     sun.add_body(uranus)
    #     sun.add_body(neptune)
    #     return sol

    # def _solarSystem_():
    #     class solarSystem(System):
    #         pass
    #     sol = solarSystem("Solar system")
    #     sun = preset_bodies.Sun
    #     mercury = preset_bodies.Mercury
    #     venus = preset_bodies.Venus
    #     earth = preset_bodies.Earth
    #     mars = preset_bodies.Mars
    #     jupiter = preset_bodies.Jupiter
    #     saturn = preset_bodies.Saturn
    #     uranus = preset_bodies.Uranus
    #     neptune = preset_bodies.Neptune
    #     mercury.orbit = preset_orbits.Mercury
    #     venus.orbit = preset_orbits.Venus
    #     earth.orbit = preset_orbits.Earth
    #     mars.orbit = preset_orbits.Mars
    #     jupiter.orbit = preset_orbits.Jupiter
    #     saturn.orbit = preset_orbits.Saturn
    #     uranus.orbit = preset_orbits.Uranus
    #     neptune.orbit = preset_orbits.Neptune
    #     sol.add_body(sun)
    #     sun.add_body(mercury)
    #     sun.add_body(venus)
    #     sun.add_body(earth)
    #     sun.add_body(mars)
    #     sun.add_body(jupiter)
    #     sun.add_body(saturn)
    #     sun.add_body(uranus)
    #     sun.add_body(neptune)
    #     return sol

    earthMoon = _earthMoon_()
    innerPlanets = _innerPlanets_()
    # outerPlanets = _outerPlanets_()
    # solarSystem = _solarSystem_()
