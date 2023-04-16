from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from logging import Logger, getLogger
from math import pi
from typing import Any, List, Type

import astropy.constants as const
import astropy.units as u
import numpy as np

from common import expand, flatten

from .orbits import Orbit


@dataclass
class _bodycontainer_:
    name: str

    def __str__(self) -> str:
        return f"{self.__class__.__name__} '{self.name.capitalize()}'" + (
            f", {self.orbit}" if hasattr(self, "_orbit_") else ""
        )

    def has_body(self, body: Type[body]) -> bool:
        return body in list(self.flattened)

    def add_body(self, body: Type[body]) -> None:
        if not self.has_body(body):
            body.parent = self
            self.bodies += [body]
            self.logger.debug(f"added {body}")

    def remove_body(self, body: Type[body]) -> None:
        if self.has_body(body):
            body.parent = None
            self.bodies = self.bodies.remove(body)
            self.logger.debug(f"removed {body}")

    @cached_property
    def logger(self) -> Logger:
        return getLogger("system").getChild(self._name)

    @property
    def parent(self) -> Type[_bodycontainer_]:
        return self._parent_ if hasattr(self, "_parent_") else None

    @parent.setter
    def parent(self, _parent_: Type[_bodycontainer_]) -> None:
        self._parent_ = _parent_

    @property
    def bodies(self) -> List[Type[_bodycontainer_]]:
        return self._bodies_ if hasattr(self, "_bodies_") else []

    @bodies.setter
    def bodies(self, _bodies_: List[Type[_bodycontainer_]]) -> None:
        self._bodies_ = _bodies_

    @property
    def flattened(self) -> List[Type[body]]:
        return list(flatten(self.expanded))

    @property
    def expanded(self) -> List[Any | Type[_bodycontainer_]]:
        return expand(self, "_bodies_")

    @cached_property
    def _name(self) -> str:
        return self.name.lower().strip().replace(" ", "-")


@dataclass
class body(_bodycontainer_):
    name: str
    mass: float
    # equatorial radius, or tuple of smallest equitorial, largest equitorial, polar
    radius: float | tuple[float, float] | tuple[float, float, float]
    rotation_period: float = 0 * u.s
    age: float = None
    _flattening_: float = None

    @property
    def flattening(self) -> u.Quantity[u.dimensionless_unscaled]:
        if hasattr(self, "_flattening_"):
            return self._flattening_
        elif isinstance(self.radius, (tuple, list)):
            a = (
                self.radius[0]
                if len(self.radius) == 2
                else 0.5 * (self.radius[0] + self.radius[1])
            )
            return (a - self.radius[-1]) / a
        return 0

    @flattening.setter
    def flattening(self, _flattening_: u.Quantity[u.dimensionless_unscaled]) -> None:
        if isinstance(self.radius, (tuple, list)):
            a = (
                self.radius[0]
                if len(self.radius) == 2
                else 0.5 * (self.radius[0] + self.radius[1])
            )
            assert (a - self.radius[-1]) / a == _flattening_
        self._flattening_ = _flattening_

    @property
    def volume(self) -> u.Quantity[u.m**3]:
        if not isinstance(self.radius, (tuple, list)):
            return 4 * pi * self.radius**3 / 3
        if len(self.radius) == 2:
            return 4 * pi * self.radius[0] ** 2 * self.radius[1] / 3
        return 4 * pi * self.radius[0] * self.radius[1] * self.radius[2] / 3

    def radius_at(self, latitude: u.Quantity[u.deg]) -> u.Quantity[u.m]:
        """Compute the radius at a given lattiude, given the flattening of the body"""
        if self.flattening and not isinstance(self.radius, (tuple, list)):
            a, b = self.radius, self.radius * (1 - self.flattening)
        elif self.flattening:
            a = (
                self.radius[0]
                if len(self.radius) == 2
                else 0.5 * (self.radius[0] + self.radius[1])
            )
            b = self.radius[-1]
        return (a**2 * np.cos(latitude) ** 2 + b**2 * np.sin(latitude)) ** 0.5

    @property
    def _radius(self) -> u.Quantity[u.m]:
        return (
            self.radius
            if not isinstance(self.radius, (tuple, list))
            else sum(self.radius) / 3
        )

    @property
    def density(self) -> u.Quantity[u.kg / u.m**3]:
        return self.mass / self.volume

    @property
    def surface_gravity(self) -> u.Quantity[u.m / u.s**2]:
        """Gravitational acceleration at the average surface radius."""
        return const.G * self.mass / self._radius**2

    @property
    def adjusted_surface_gravity(
        self, latitude: u.Quantity[u.deg]
    ) -> u.Quantity[u.m / u.s**2]:
        """Gravitation acceleration, adjusted for
        rotation of the body at a given latitude."""
        r = self.radius_at(latitude)
        _r = 2 * np.pi * np.sin(90 * u.degree - latitude) * r
        v = _r / self.rotation_period if self.rotation_period else 0
        return self.surface_gravity - v**2 / r

    @property
    def surface_escape_velocity(self) -> u.Quantity[u.m / u.s]:
        return (2 * const.G * self.mass / self._radius) ** 0.5

    @property
    def orbit(self) -> Orbit:
        return self._orbit_ if hasattr(self, "_orbit_") else None

    @orbit.setter
    def orbit(self, _orbit_: Orbit) -> None:
        self._orbit_ = _orbit_
        self._orbit_.body = self

    @property
    def pos(self) -> tuple[u.Quantity[u.m], u.Quantity[u.m], u.Quantity[u.m]]:
        return self.position

    @property
    def position(self) -> tuple[u.Quantity[u.m], u.Quantity[u.m], u.Quantity[u.m]]:
        if self.orbit:
            return self.orbit.to_cartesian.position
        return getattr(self, "_position_", (0, 0, 0) * u.m)


@dataclass
class star(body):
    luminosity: u.Quantity[u.W] = 0 * u.W
    metallicity: u.Quantity[u.dimensionless_unscaled] = 0

    @property
    def magnitude(self) -> u.Quantity[u.dimensionless_unscaled]:
        """Bolometric magnitude of the star"""
        return -2.5 * np.log10(self.luminosity / const.L_bol0)

    @property
    def lifetime(self) -> u.Quantity[u.s]:
        """Predicted lifetime length"""
        return (10e10 * u.year * (self.mass / const.M_sun) ** 2.5).to(u.s)

    @property
    def temperature(self) -> u.Quantity[u.K]:
        """Expected surface temperature"""
        return (
            (self.luminosity / (4 * np.pi * self._radius**2 * const.sigma_sb)) ** 0.25
        ).to(u.K)


@dataclass
class planet(body):
    pass


@dataclass
class moon(planet):
    pass


class presets:
    """Define some pre-set body objects for testing"""

    Sun = star(
        "Sun",
        mass=const.M_sun,
        radius=const.R_sun,
        luminosity=const.L_sun,
        metallicity=0.0122,
        age=4.57e9 * u.year,
        rotation_period=25.05 * u.day,
        _flattening_=9e-6,
    )
    Mercury = planet(
        "Mercury",
        mass=3.3011e23 * u.kg,
        radius=2439.7 * u.km,
        rotation_period=176 * u.day,
        age=Sun.age,
        _flattening_=0.0009,
    )
    Venus = planet(
        "Venus",
        mass=4.8675e24 * u.kg,
        radius=6051.8 * u.km,
        rotation_period=-243.0226 * u.day,
        age=Sun.age,
    )
    Earth = planet(
        "Earth",
        mass=5.972168e24 * u.kg,
        radius=(6378.137, 6356.752) * u.km,
        rotation_period=0.99726968 * u.day,
        age=Sun.age,
    )
    Moon = moon(
        "Moon",
        mass=7.342e22 * u.kg,
        radius=(1738.1, 1736) * u.km,
        rotation_period=27.321661 * u.day,
        age=4.452e9 * u.year,
    )
    Mars = planet(
        "Mars",
        mass=6.4171e23 * u.kg,
        radius=(3396.2, 3396.2) * u.km,
        rotation_period=1.025957 * u.day,
        age=Sun.age,
    )
    Jupiter = planet(
        "Jupiter",
        mass=1.8982e27 * u.kg,
        radius=(71492, 66854) * u.km,
        rotation_period=9.9250 * u.hour,
        age=Sun.age,
    )
    Saturn = planet(
        "Saturn",
        mass=5.6834e26 * u.kg,
        radius=(60268, 54364) * u.km,
        rotation_period=10.5433 * u.hour,
        age=Sun.age,
    )
    Uranus = planet(
        "Uranus",
        mass=8.6810e25 * u.kg,
        radius=(25559, 24973) * u.kg,
        rotation_period=-0.71833 * u.day,
        age=Sun.age,
    )
    Neptune = planet(
        "Neptune",
        mass=1.02413e26 * u.kg,
        radius=(24764, 24341) * u.km,
        rotation_period=0.6713 * u.day,
        age=Sun.age,
    )
