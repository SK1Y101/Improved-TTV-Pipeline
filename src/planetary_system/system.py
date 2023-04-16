from dataclasses import dataclass
from logging import getLogger
from typing import Any, Type

import astropy.constants as const
import astropy.units as u

from common import log_children

from .bodies import _bodycontainer_
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
        self._plotter_ = plotter(self)

    def plot_system(self, style: str) -> None:
        self.plotter.plot_system(style=style)

    def fetch_body(self, name: str) -> Type[_bodycontainer_]:
        return [body for body in self.flattened[1:] if body.name == name][0]

    @property
    def total_mass(self) -> u.Quantity[u.kg]:
        return sum(body.mass for body in self.flattened[1:])

    @property
    def centre_of_mass(
        self,
    ) -> tuple[u.Quantity[u.m], u.Quantity[u.m], u.Quantity[u.m]]:
        pos = (0, 0, 0) * u.m * u.kg
        for body in self.flattened[1:]:
            print(body)
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
        for body in self.flattened[1:]:
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


class System(_system_):
    pass


class presets:
    def _solarSystem_():
        sol = System("Solar system")
        sun = preset_bodies.Sun
        mercury = preset_bodies.Mercury
        venus = preset_bodies.Venus
        earth = preset_bodies.Earth
        moon = preset_bodies.Moon
        mars = preset_bodies.Mars
        jupiter = preset_bodies.Jupiter
        saturn = preset_bodies.Saturn
        uranus = preset_bodies.Uranus
        neptune = preset_bodies.Neptune
        mercury.orbit = preset_orbits.Mercury
        venus.orbit = preset_orbits.Venus
        earth.orbit = preset_orbits.Earth
        moon.orbit = preset_orbits.Moon
        mars.orbit = preset_orbits.Mars
        jupiter.orbit = preset_orbits.Jupiter
        saturn.orbit = preset_orbits.Saturn
        uranus.orbit = preset_orbits.Uranus
        neptune.orbit = preset_orbits.Neptune
        sol.add_body(sun)
        sun.add_body(mercury)
        sun.add_body(venus)
        earth.add_body(moon)
        sun.add_body(earth)
        sun.add_body(mars)
        sun.add_body(jupiter)
        sun.add_body(saturn)
        sun.add_body(uranus)
        sun.add_body(neptune)
        return sol

    def _extendedSolarSystem_(_solarSystem_):
        sol = _solarSystem_()
        return sol

    solarSystem = _solarSystem_()
    extendedSolarSystem = _extendedSolarSystem_(_solarSystem_)
