from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import astropy.constants as const
import astropy.units as u
import numpy as np


def from_cartesian(
    pos: tuple[u.Quantity[u.m], u.Quantity[u.m], u.Quantity[u.m]],
    vel: tuple[u.Quantity[u.m / u.s], u.Quantity[u.m / u.s], u.Quantity[u.m / u.s]],
) -> Orbit:
    return Orbit()


@dataclass
class cartesian_vector:
    position: tuple[u.Quantity[u.m], u.Quantity[u.m], u.Quantity[u.m]] = (0, 0, 0) * u.m
    velocity: tuple[
        u.Quantity[u.m / u.s], u.Quantity[u.m / u.s], u.Quantity[u.m / u.s]
    ] = ((0, 0, 0) * u.m / u.s)


@dataclass
class Orbit:
    semimajor_axis: u.Quantity[u.m]
    eccentricity: u.Quantity[u.dimensionless_unscaled] = 0
    inclination: u.Quantity[u.deg] = 0
    mean_anomaly: u.Quantity[u.deg] = 0
    ascending_longitude: u.Quantity[u.deg] = 0
    argument_periapse: u.Quantity[u.deg] = 0

    @property
    def a(self) -> u.Quantity[u.m]:
        """Semi-major Axis"""
        return self.semimajor_axis

    @property
    def e(self) -> u.Quantity[u.dimensionless_unscaled]:
        """Eccentricity"""
        return self.eccentricity

    @property
    def i(self) -> u.Quantity[u.deg]:
        """Inclination"""
        return self.inclination

    @property
    def M(self) -> u.Quantity[u.deg]:
        """Mean Anomaly"""
        return self.mean_anomaly

    @property
    def theta(self) -> u.Quantity[u.deg]:
        """True Anomaly"""
        return self.true_anomaly

    @property
    def Omega(self) -> u.Quantity[u.deg]:
        """Longitude of the Ascending Node"""
        return self.ascending_longitude

    @property
    def omega(self) -> u.Quantity[u.deg]:
        """Argument of Periapse"""
        return self.argument_periapse

    @property
    def p(self) -> u.Quantity[u.m]:
        """Semi-latis Rectum"""
        return (self.a * (1 - self.e) ** 2).to(u.m)

    def __str__(self) -> str:
        return f"Orbiting at {self.r}"

    @property
    def body(self) -> Any:
        if hasattr(self, "_body_"):
            return self._body_

    @body.setter
    def body(self, _body_: Any) -> None:
        self._body_ = _body_

    def attached(self) -> bool:
        if not self.body:
            raise Exception(f"Orbit requires a body: {self} not attached to one.")
        elif not self.body.parent:
            raise Exception(
                f"Orbit requires a parent: {self.body.name} does not have a parent"
            )
        return True

    @property
    def b(self) -> u.Quantity[u.m]:
        """Semi-major Axis"""
        return self.semiminor_axis

    @property
    def semiminor_axis(self) -> u.Quantity[u.m]:
        return (self.a * (1 - self.e**2) ** 0.5).to(u.m)

    @property
    def true_anomaly(self) -> u.Quantity[u.deg]:
        M = self.M
        sinM = [np.sin(i * M) for i in range(0, 7)]
        e = [self.e**i for i in range(0, 7)]
        return (
            M
            + 2 * e[1] * sinM[1] * u.deg
            + (5 / 4) * e[2] * sinM[2] * u.deg
            + (e[3] / 12) * (13 * sinM[3] - 3 * sinM[1]) * u.deg
            + (e[4] / 96) * (103 * sinM[4] - 44 * sinM[2]) * u.deg
            + (e[5] / 960) * (1097 * sinM[5] - 645 * sinM[3] + 50 * sinM[1]) * u.deg
            + (e[6] / 960) * (1223 * sinM[6] - 902 * sinM[4] + 85 * sinM[2]) * u.deg
        ).to(u.deg)

    @property
    def mu(self) -> u.Quantity[u.m**3 / u.s**2]:
        """Gravitational parameter"""
        self.attached()
        return (const.G * (self.body.mass + self.body.parent.mass)).to(
            u.m**3 / u.s**2
        )

    @property
    def radius(self) -> u.Quantity[u.m]:
        return (self.a * (1 - self.e**2) / (1 + self.e * np.cos(self.theta))).to(u.m)

    @property
    def r(self) -> u.Quantity[u.m]:
        """Radius of Current Position"""
        return self.radius

    def velocity(self, radius: float = None) -> u.Quantity[u.m / u.s]:
        self.attached()
        if not radius:
            radius = self.r
        return ((self.mu * (2 / radius - 1 / self.a)) ** 0.5).to(u.m / u.s)

    def v(self, radius: float = None) -> float:
        """Velocity at current or defined position"""
        return self.velocity(radius)

    @property
    def period(self) -> u.Quantity[u.s]:
        self.attached()
        return (2 * np.pi * (self.a**3 / self.mu) ** 0.5).to(u.s)

    def T(self) -> float:
        """Orbital Period"""
        return self.period

    @property
    def to_cartesian(self) -> cartesian_vector:
        print(self.radius)
        pos_vec = np.array([self.radius.value, 0, 0]) * u.m
        vel_vec = (
            np.array(
                [
                    0,
                    (
                        (self.mu / self.p) ** 0.5 * (1 + self.e * np.sin(self.theta))
                    ).value,
                    0,
                ]
            )
            * u.m
            / u.s
        )
        rot_mat_i = np.array(
            [
                [1, 0, 0],
                [0, np.cos(self.i), np.sin(self.i)],
                [0, -np.sin(self.i), np.cos(self.i)],
            ]
        )
        rot_mat_o = np.array(
            [
                [np.cos(self.omega), -np.sin(self.omega), 0],
                [np.sin(self.omega), np.cos(self.omega), 0],
                [0, 0, 1],
            ]
        )
        rot_mat_O = np.array(
            [
                [np.cos(self.Omega), -np.sin(self.Omega), 0],
                [np.sin(self.Omega), np.cos(self.Omega), 0],
                [0, 0, 1],
            ]
        )
        rot_mat = rot_mat_O * rot_mat_i * rot_mat_o
        return cartesian_vector(rot_mat * pos_vec, rot_mat * vel_vec)


class presets:
    Mercury = Orbit(
        semimajor_axis=0.387098 * const.au,
        eccentricity=0.205630,
        inclination=7.005 * u.deg,
        mean_anomaly=174.796 * u.deg,
        ascending_longitude=48.33 * u.deg,
        argument_periapse=29.124 * u.deg,
    )
    Venus = Orbit(
        semimajor_axis=0.387098 * const.au,
        eccentricity=0.205630,
        inclination=7.005 * u.deg,
        mean_anomaly=174.796 * u.deg,
        ascending_longitude=48.33 * u.deg,
        argument_periapse=29.124 * u.deg,
    )
    Earth = Orbit(
        semimajor_axis=0.387098 * const.au,
        eccentricity=0.205630,
        inclination=7.005 * u.deg,
        mean_anomaly=174.796 * u.deg,
        ascending_longitude=48.33 * u.deg,
        argument_periapse=29.124 * u.deg,
    )
    Moon = Orbit(
        semimajor_axis=0.387098 * const.au,
        eccentricity=0.205630,
        inclination=7.005 * u.deg,
        mean_anomaly=174.796 * u.deg,
        ascending_longitude=48.33 * u.deg,
        argument_periapse=29.124 * u.deg,
    )
    Mars = Orbit(
        semimajor_axis=0.387098 * const.au,
        eccentricity=0.205630,
        inclination=7.005 * u.deg,
        mean_anomaly=174.796 * u.deg,
        ascending_longitude=48.33 * u.deg,
        argument_periapse=29.124 * u.deg,
    )
    Jupiter = Orbit(
        semimajor_axis=0.387098 * const.au,
        eccentricity=0.205630,
        inclination=7.005 * u.deg,
        mean_anomaly=174.796 * u.deg,
        ascending_longitude=48.33 * u.deg,
        argument_periapse=29.124 * u.deg,
    )
    Saturn = Orbit(
        semimajor_axis=0.387098 * const.au,
        eccentricity=0.205630,
        inclination=7.005 * u.deg,
        mean_anomaly=174.796 * u.deg,
        ascending_longitude=48.33 * u.deg,
        argument_periapse=29.124 * u.deg,
    )
    Uranus = Orbit(
        semimajor_axis=0.387098 * const.au,
        eccentricity=0.205630,
        inclination=7.005 * u.deg,
        mean_anomaly=174.796 * u.deg,
        ascending_longitude=48.33 * u.deg,
        argument_periapse=29.124 * u.deg,
    )
    Neptune = Orbit(
        semimajor_axis=0.387098 * const.au,
        eccentricity=0.205630,
        inclination=7.005 * u.deg,
        mean_anomaly=174.796 * u.deg,
        ascending_longitude=48.33 * u.deg,
        argument_periapse=29.124 * u.deg,
    )
