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

empty_3_vector = np.zeros((3, 1))


@dataclass
class _bodycontainer_:
    name: str

    def __str__(self) -> str:
        return f"{self.__class__.__name__} '{self.name.capitalize()}'" + (
            f", {self.orbit}" if hasattr(self, "_orbit_") else ""
        )

    def has_body(self, body: Type[Body]) -> bool:
        return body in list(self.flattened)

    def add_body(self, body: Type[Body]) -> None:
        if not self.has_body(body):
            body.parent = self
            self.bodies += [body]
            self.logger.debug(f"added {body}")

    def remove_body(self, body: Type[Body]) -> None:
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
    def flattened(self) -> List[Type[Body]]:
        return list(flatten(self.expanded))[1:]

    @property
    def expanded(self) -> List[Any | Type[_bodycontainer_]]:
        return expand(self, "_bodies_")

    @cached_property
    def _name(self) -> str:
        return self.name.lower().strip().replace(" ", "-")


@dataclass
class Body(_bodycontainer_):
    name: str
    mass: float
    # equatorial radius, or tuple of smallest equitorial, largest equitorial, polar
    radius: float | tuple[float, float] | tuple[float, float, float]
    rotation_period: float = 0 * u.s
    age: float = None
    _flattening_: float = None
    _symbol_: str = None

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
    def orbit(self) -> Orbit | None:
        return self._orbit_ if hasattr(self, "_orbit_") else None

    @orbit.setter
    def orbit(self, _orbit_: Orbit) -> None:
        _orbit_.body = self
        self._orbit_ = _orbit_

    @property
    def pos(self) -> tuple[u.Quantity[u.m], u.Quantity[u.m], u.Quantity[u.m]]:
        return self.position

    @property
    def position(self) -> tuple[u.Quantity[u.m], u.Quantity[u.m], u.Quantity[u.m]]:
        if self.orbit:
            return self.orbit.position
        return getattr(self, "_position_", empty_3_vector * u.m)

    @position.setter
    def position(
        self, pos: tuple[u.Quantity[u.m], u.Quantity[u.m], u.Quantity[u.m]]
    ) -> None:
        if self.orbit:
            self.orbit.position = pos
        else:
            self._position_ = pos

    @property
    def abs_pos(self) -> tuple[u.Quantity[u.m], u.Quantity[u.m], u.Quantity[u.m]]:
        return self.absolute_position

    @property
    def absolute_position(
        self,
    ) -> tuple[u.Quantity[u.m], u.Quantity[u.m], u.Quantity[u.m]]:
        if self.parent and hasattr(self.parent, "absolute_position"):
            _x, _y, _z = self.parent.absolute_position
            x, y, z = self.position
            return x + _x, y + _y, z + _z
        return self.position


@dataclass
class Star(Body):
    luminosity: u.Quantity[u.W] = 0 * u.W
    metallicity: u.Quantity[u.dimensionless_unscaled] = 0
    _symbol_: str = "★"

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

    @property
    def colour(self) -> tuple[float, float, float]:
        """Colour of a star according to it's temperature,
        returns an rgb tuple in the range [0-255]"""

        def _interp_(
            low: float, high: float, _min: float, _max: float, current: float
        ) -> float:
            if current <= _min:
                return low
            elif current >= _max:
                return high
            return low + (current - _min) * (high - low) / (_max - _min)

        temp = self.temperature.value
        # TODO: Better than this
        if temp < 3500:
            return (0xFF, _interp_(0x00, 0xA5, 1000, 3500, temp), 0x0)
        elif temp < 5000:
            return (0xFF, _interp_(0xA5, 0xFF, 3500, 5000, temp), 0x0)
        elif temp < 6000:
            return (0xFF, 0xFF, _interp_(0, 0xFF, 5000, 6000, temp))
        elif temp < 7500:
            return (
                _interp_(0xFF, 0xAD, 6000, 7500, temp),
                _interp_(0xFF, 0xD8, 6000, 7500, temp),
                _interp_(0xFF, 0xE6, 6000, 7500, temp),
            )
        return (
            _interp_(0xAD, 0x00, 7500, 10000, temp),
            _interp_(0xD8, 0x00, 7500, 10000, temp),
            _interp_(0xE6, 0xFF, 7500, 10000, temp),
        )

    @property
    def normalised_colour(self) -> tuple[float, float, float]:
        """Colour of a star according to it's temperature,
        returns an rgb tuple in the range [0-1]"""
        r, g, b = self.colour
        return (r / 255, g / 255, b / 255)


@dataclass
class Planet(Body):
    pass


@dataclass
class Moon(Planet):
    _symbol_: str = "🌒︎"


class presets:
    """Define some pre-set body objects for testing"""

    Sun = Star(
        "Sun",
        mass=const.M_sun,
        radius=const.R_sun,
        luminosity=const.L_sun,
        metallicity=0.0122,
        age=4.57e9 * u.year,
        rotation_period=25.05 * u.day,
        _flattening_=9e-6,
        _symbol_="☉",
    )
    Mercury = Planet(
        "Mercury",
        mass=3.3011e23 * u.kg,
        radius=2439.7 * u.km,
        rotation_period=176 * u.day,
        age=Sun.age,
        _flattening_=0.0009,
        _symbol_="☿",
    )
    Venus = Planet(
        "Venus",
        mass=4.8675e24 * u.kg,
        radius=6051.8 * u.km,
        rotation_period=-243.0226 * u.day,
        age=Sun.age,
        _symbol_="♀",
    )
    Earth = Planet(
        "Earth",
        mass=5.972168e24 * u.kg,
        radius=(6378.137, 6356.752) * u.km,
        rotation_period=0.99726968 * u.day,
        age=Sun.age,
        _symbol_="♁",
    )
    Moon = Moon(
        "Moon",
        mass=7.342e22 * u.kg,
        radius=(1738.1, 1736) * u.km,
        rotation_period=27.321661 * u.day,
        age=4.452e9 * u.year,
    )
    Mars = Planet(
        "Mars",
        mass=6.4171e23 * u.kg,
        radius=(3396.2, 3396.2) * u.km,
        rotation_period=1.025957 * u.day,
        age=Sun.age,
        _symbol_="♂",
    )
    Jupiter = Planet(
        "Jupiter",
        mass=1.8982e27 * u.kg,
        radius=(71492, 66854) * u.km,
        rotation_period=9.9250 * u.hour,
        age=Sun.age,
        _symbol_="♃",
    )
    Saturn = Planet(
        "Saturn",
        mass=5.6834e26 * u.kg,
        radius=(60268, 54364) * u.km,
        rotation_period=10.5433 * u.hour,
        age=Sun.age,
        _symbol_="♄",
    )
    Uranus = Planet(
        "Uranus",
        mass=8.6810e25 * u.kg,
        radius=(25559, 24973) * u.kg,
        rotation_period=-0.71833 * u.day,
        age=Sun.age,
        _symbol_="⛢",
    )
    Neptune = Planet(
        "Neptune",
        mass=1.02413e26 * u.kg,
        radius=(24764, 24341) * u.km,
        rotation_period=0.6713 * u.day,
        age=Sun.age,
        _symbol_="♆",
    )
