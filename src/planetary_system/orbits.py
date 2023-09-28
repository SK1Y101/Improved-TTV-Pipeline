from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import astropy.constants as const
import astropy.units as u
import numpy as np


def to_cartesian(keplerian: kepler_vector) -> cartesian_vector:
    """https://downloads.rene-schwarz.com/download/M001-Keplerian_Orbit_Elements_to_Cartesian_State_Vectors.pdf"""
    V = keplerian.theta
    E = np.arctan2((1 - keplerian.e**2) ** 0.5 * np.sin(V), keplerian.e * np.cos(V))
    r = keplerian.a * (1 - keplerian.e**2) / (1 + keplerian.e * np.cos(V))
    # position and velocity vectors in orbital frame
    _p = r * np.array([[np.cos(V)], [np.sin(V)]])
    _v = (
        (keplerian.mu * keplerian.a) ** 0.5
        / r
        * np.array([[-np.sin(E)], [(1 - keplerian.e**2) ** 0.5 * np.cos(E)]])
    )
    # sine, cosine shorthand
    sininc, cosinc = np.sin(keplerian.i), np.cos(keplerian.i)
    sinlan, coslan = np.sin(keplerian.Omega), np.cos(keplerian.Omega)
    sinarg, cosarg = np.sin(keplerian.omega), np.cos(keplerian.omega)
    # rotation matrix
    rx1 = cosarg * coslan - sinarg * cosinc * sinlan
    rx2 = sinarg * coslan + cosarg * cosinc * sinlan
    ry1 = cosarg * sinlan + sinarg * cosinc * coslan
    ry2 = cosarg * cosinc * coslan - sinarg * sinlan
    rz1 = sinarg * sininc
    rz2 = cosarg * sininc
    # position in inertial
    x, y, z = (
        _p[0] * rx1 - _p[1] * rx2,
        _p[0] * ry1 + _p[1] * ry2,
        _p[0] * rz1 + _p[1] * rz2,
    )
    # velocity in inertial
    vx, vy, vz = (
        _v[0] * rx1 - _v[1] * rx2,
        _v[0] * ry1 + _v[1] * ry2,
        _v[0] * rz1 + _v[1] * rz2,
    )
    return cartesian_vector(
        x.to(u.m),
        y.to(u.m),
        z.to(u.m),
        vx.to(u.m / u.s),
        vy.to(u.m / u.s),
        vz.to(u.m / u.s),
    )


def to_kepler(cartesian: cartesian_vector) -> kepler_vector:
    """https://downloads.rene-schwarz.com/download/M002-Cartesian_State_Vectors_to_Keplerian_Orbit_Elements.pdf"""
    tau = 2 * np.pi * u.rad
    # position and velocity
    R = cartesian.position.T
    V = cartesian.velocity.T
    # angular momentum
    h = np.cross(R, V)
    # eccentricity
    e = np.cross(V, h) / cartesian.mu - R / np.linalg.norm(R)
    _e = np.linalg.norm(e)
    # true anomaly
    n = np.cross(np.array([0, 0, 1]).T, h)
    v = np.arccos(np.dot(e, R.T) / (np.linalg.norm(e) * np.linalg.norm(R)))
    v = v if np.dot(R, V.T) >= 0 else tau - v
    # inclination
    i = np.arccos(h.T[2] / np.linalg.norm(h))
    # eccentric anomaly
    E = 2 * np.arctan2(np.tan(v / 2), ((1 + _e) / (1 - _e)) ** 0.5)
    # longitude of ascending node
    Omega = np.arccos(n.T[0] / np.linalg.norm(n))
    Omega = Omega if n.T[1] >= 0 else tau - Omega
    # argument of periapse
    omega = np.arccos(np.dot(n, e.T) / (np.linalg.norm(n) * np.linalg.norm(e)))
    omega = omega if n.T[2] >= 0 else tau - omega
    # mean anomaly
    M = E - _e * np.sin(E) * u.rad
    # semimajor axis
    a = 1 / (2 / np.linalg.norm(R) - np.linalg.norm(V) ** 2 / cartesian.mu)
    return kepler_vector(a=a, e=_e, i=i, M=M, omega=omega, Omega=Omega)


@dataclass
class cartesian_vector:
    x: u.Quantity[u.m] = 0 * u.m
    y: u.Quantity[u.m] = 0 * u.m
    z: u.Quantity[u.m] = 0 * u.m
    vx: u.Quantity[u.m / u.s] = 0 * u.m / u.s
    vy: u.Quantity[u.m / u.s] = 0 * u.m / u.s
    vz: u.Quantity[u.m / u.s] = 0 * u.m / u.s

    @property
    def a(self) -> u.Quantity[u.m]:
        return self.kepler.a

    @property
    def e(self) -> u.Quantity[u.dimensionless_unscaled]:
        return self.kepler.e

    @property
    def i(self) -> u.Quantity[u.deg]:
        return self.kepler.i

    @property
    def M(self) -> u.Quantity[u.deg]:
        return self.kepler.M

    @property
    def omega(self) -> u.Quantity[u.deg]:
        return self.kepler.omega

    @property
    def Omega(self) -> u.Quantity[u.deg]:
        return self.kepler.Omega

    @property
    def theta(self) -> u.Quantity[u.deg]:
        return self.kepler.theta

    @property
    def kepler(self) -> kepler_vector:
        kepler = to_kepler(self)
        kepler.orbit = self.orbit
        return kepler

    @property
    def mu(self) -> u.Quantity[u.m**3 / u.s**2]:
        """Gravitational parameter"""
        if hasattr(self, "_orbit_"):
            return self.orbit.mu
        raise Exception(
            "An orbiting body is required to define the gravitational parameter"
        )

    @property
    def position(self) -> Tuple[u.Quantity[u.m], u.Quantity[u.m], u.Quantity[u.m]]:
        return (self.x.to(u.m).value, self.y.to(u.m).value, self.z.to(u.m).value) * u.m

    @position.setter
    def position(
        self, pos: Tuple[u.Quantity[u.m], u.Quantity[u.m], u.Quantity[u.m]]
    ) -> None:
        self.x, self.y, self.z = pos.to(u.m)

    @property
    def velocity(
        self,
    ) -> Tuple[u.Quantity[u.m / u.s], u.Quantity[u.m / u.s], u.Quantity[u.m / u.s]]:
        return (
            (
                self.vx.to(u.m / u.s).value,
                self.vy.to(u.m / u.s).value,
                self.vz.to(u.m / u.s).value,
            )
            * u.m
            / u.s
        )

    @property
    def line_element(self) -> str:
        return self.orbit.line_element


@dataclass
class kepler_vector:
    a: u.Quantity[u.m] = 0 * u.m
    e: u.Quantity[u.dimensionless_unscaled] = 0 * u.dimensionless_unscaled
    i: u.Quantity[u.deg] = 0 * u.deg
    M: u.Quantity[u.deg] = 0 * u.deg
    omega: u.Quantity[u.deg] = 0 * u.deg
    Omega: u.Quantity[u.deg] = 0 * u.deg

    @property
    def x(self) -> u.Quantity[u.m]:
        return self.cartesian.x

    @property
    def y(self) -> u.Quantity[u.m]:
        return self.cartesian.y

    @property
    def z(self) -> u.Quantity[u.m]:
        return self.cartesian.z

    @property
    def vx(self) -> u.Quantity[u.m / u.s]:
        return self.cartesian.vx

    @property
    def vy(self) -> u.Quantity[u.m / u.s]:
        return self.cartesian.vy

    @property
    def vz(self) -> u.Quantity[u.m / u.s]:
        return self.cartesian.vz

    @property
    def theta(self) -> u.Quantity[u.deg]:
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
    def position(self) -> Tuple[u.Quantity[u.m], u.Quantity[u.m], u.Quantity[u.m]]:
        return self.cartesian.position

    @property
    def velocity(
        self,
    ) -> Tuple[u.Quantity[u.m / u.s], u.Quantity[u.m / u.s], u.Quantity[u.m / u.s]]:
        return self.cartesian.velocity

    @property
    def cartesian(self) -> cartesian_vector:
        cartesian = to_cartesian(self)
        cartesian.orbit = self.orbit
        return cartesian

    @property
    def mu(self) -> u.Quantity[u.m**3 / u.s**2]:
        """Gravitational parameter"""
        return self.orbit.mu

    @property
    def line_element(self) -> str:
        return self.orbit.line_element


@dataclass
class Orbit:
    semimajor_axis: u.Quantity[u.m]
    eccentricity: u.Quantity[u.dimensionless_unscaled] = 0
    inclination: u.Quantity[u.deg] = 0
    mean_anomaly: u.Quantity[u.deg] = 0
    ascending_longitude: u.Quantity[u.deg] = 0
    argument_periapse: u.Quantity[u.deg] = 0

    def update(
        self,
        a: u.Quantity[u.m],
        e: u.Quantity[u.dimensionless_unscaled],
        i: u.Quantity[u.deg],
        M: u.Quantity[u.deg],
        omega: u.Quantity[u.deg],
        Omega: u.Quantity[u.deg],
    ) -> None:
        self.semimajor_axis = a.to(u.m)
        self.eccentricity = e
        self.inclination = i.to(u.deg)
        self.mean_anomaly = M.to(u.deg)
        self.ascending_longitude = Omega.to(u.deg)
        self.argument_periapse = omega.to(u.deg)

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
        return self.line_element

    @property
    def line_element(self) -> str:
        """An approximation of TLE, useful for comparing orbits.
        Example:
        SMA-m-ld   inc-deg  Omeg-deg ecc-ld  omeg-deg Mean-deg
        7789081+07 093.0114 037.6781 0067807 008.3428 012.5737

        Note: Semimajor axis and eccentricity have both had leading
        zeros removed: the above example should be read:
        SMA: 0.7789081 * 10^7
        ECC: 0.0067807
        """
        a = self.a.to(u.m).value
        p = 10 ** np.ceil(np.log10(a))
        return " ".join(
            [
                f"{f'{a/p:.7f}'[2:]}{f'{a*10:e}'[-3:]}",
                f"{self.i.to(u.deg).value % 360:08.4f}",
                f"{self.Omega.to(u.deg).value % 360:08.4f}",
                f"{self.e.value:.7f}"[2:],
                f"{self.omega.to(u.deg).value % 360:08.4f}",
                f"{self.M.to(u.deg).value % 360:08.4f}",
            ]
        )

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

    def prograde_velocity(
        self, radius: u.Quantity[u.m] = None
    ) -> u.Quantity[u.m / u.s]:
        self.attached()
        if not radius:
            radius = self.r
        return ((self.mu * (2 / radius - 1 / self.a)) ** 0.5).to(u.m / u.s)

    def prograde_v(self, radius: u.Quantity[u.m] = None) -> u.Quantity[u.m / u.s]:
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
    def cartesian(self) -> cartesian_vector:
        return self.kepler.cartesian

    @property
    def kepler(self) -> kepler_vector:
        kepler = kepler_vector(
            a=self.a, e=self.e, i=self.i, M=self.M, omega=self.omega, Omega=self.Omega
        )
        kepler.orbit = self
        return kepler

    @property
    def position(self) -> Tuple[u.Quantity[u.m], u.Quantity[u.m], u.Quantity[u.m]]:
        return self.cartesian.position

    @position.setter
    def position(self, pos: Tuple[u.Quantity[u.m], u.Quantity[u.m], u.Quantity[u.m]]):
        self.cartesian.x, self.cartesian.y, self.cartesian.z = pos

    @property
    def velocity(
        self,
    ) -> Tuple[u.Quantity[u.m / u.s], u.Quantity[u.m / u.s], u.Quantity[u.m / u.s]]:
        return self.cartesian.velocity

    @property
    def x(self) -> u.Quantity[u.m]:
        return self.cartesian.x

    @property
    def y(self) -> u.Quantity[u.m]:
        return self.cartesian.y

    @property
    def z(self) -> u.Quantity[u.m]:
        return self.cartesian.z

    @property
    def vx(self) -> u.Quantity[u.m / u.s]:
        return self.cartesian.vx

    @property
    def vy(self) -> u.Quantity[u.m / u.s]:
        return self.cartesian.vy

    @property
    def vz(self) -> u.Quantity[u.m / u.s]:
        return self.cartesian.vz

    def orbit_trail(self, points: int) -> Tuple[u.Quantity[u.m], u.Quantity[u.m]]:
        theta = np.arctan2(self.x, self.y)
        psi = np.linspace(0, 1.5 * np.pi, points) * u.rad + theta
        r = self.a * (1 - self.e**2) / (1 + self.e * np.cos(psi))
        x, y = r * np.sin(psi), r * np.cos(psi)

        _x, _y, _z = self.body.absolute_position
        x_, y_, z_ = self.body.position
        x += _x - x_
        y += _y - y_
        return x, y


class presets:
    Mercury = Orbit(
        semimajor_axis=0.387098 * const.au,
        eccentricity=0.205630 * u.dimensionless_unscaled,
        inclination=7.005 * u.deg,
        mean_anomaly=174.796 * u.deg,
        ascending_longitude=48.33 * u.deg,
        argument_periapse=29.124 * u.deg,
    )
    Venus = Orbit(
        semimajor_axis=0.723332 * const.au,
        eccentricity=0.006772 * u.dimensionless_unscaled,
        inclination=3.39458 * u.deg,
        mean_anomaly=50.115 * u.deg,
        ascending_longitude=76.680 * u.deg,
        argument_periapse=54.884 * u.deg,
    )
    Earth = Orbit(
        semimajor_axis=149598023 * u.km,
        eccentricity=0.0167086 * u.dimensionless_unscaled,
        inclination=0.00005 * u.deg,
        mean_anomaly=358.617 * u.deg,
        ascending_longitude=-11.26064 * u.deg,
        argument_periapse=114.20783 * u.deg,
    )
    Moon = Orbit(
        semimajor_axis=384399 * u.km,
        eccentricity=0.0549 * u.dimensionless_unscaled,
        inclination=5.145 * u.deg,
        mean_anomaly=0 * u.deg,
        ascending_longitude=0 * u.deg,
        argument_periapse=0 * u.deg,
    )
    Mars = Orbit(
        semimajor_axis=1.52368055 * const.au,
        eccentricity=0.0934 * u.dimensionless_unscaled,
        inclination=1.850 * u.deg,
        mean_anomaly=19.412 * u.deg,
        ascending_longitude=49.57854 * u.deg,
        argument_periapse=286.5 * u.deg,
    )
    Jupiter = Orbit(
        semimajor_axis=5.2038 * const.au,
        eccentricity=0.0489 * u.dimensionless_unscaled,
        inclination=1.303 * u.deg,
        mean_anomaly=20.020 * u.deg,
        ascending_longitude=100.464 * u.deg,
        argument_periapse=273.867 * u.deg,
    )
    Saturn = Orbit(
        semimajor_axis=9.5826 * const.au,
        eccentricity=0.0565 * u.dimensionless_unscaled,
        inclination=2.485 * u.deg,
        mean_anomaly=317.020 * u.deg,
        ascending_longitude=113.665 * u.deg,
        argument_periapse=339.392 * u.deg,
    )
    Uranus = Orbit(
        semimajor_axis=19.19126 * const.au,
        eccentricity=0.04717 * u.dimensionless_unscaled,
        inclination=0.773 * u.deg,
        mean_anomaly=142.2386 * u.deg,
        ascending_longitude=74.006 * u.deg,
        argument_periapse=96.998857 * u.deg,
    )
    Neptune = Orbit(
        semimajor_axis=30.07 * const.au,
        eccentricity=0.008678 * u.dimensionless_unscaled,
        inclination=1.770 * u.deg,
        mean_anomaly=259.883 * u.deg,
        ascending_longitude=131.783 * u.deg,
        argument_periapse=273.187 * u.deg,
    )
