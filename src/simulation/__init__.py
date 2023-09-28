""" Define a way of propogating the state of the system forwards in time. """
from __future__ import annotations

from logging import Logger

import astropy.constants as const
import astropy.units as u
import rebound
import reboundx

from planetary_system.bodies import Body

# if TYPE_CHECKING:
from planetary_system.system import System


class Simulation:
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
        if _system_ and _system_.simulation is not self:
            _system_.simulation = self

    @property
    def logger(self) -> Logger:
        if self.system:
            return self.system.logger

    def base_units(
        self, sim: rebound.Simulation
    ) -> tuple[u.Quantity, u.Quantity, u.Quantity]:
        return (
            u.Unit(sim.units["mass"]),
            u.Unit(sim.units["length"]),
            u.Unit(sim.units["time"]),
        )

    @property
    def sim(self) -> rebound.Simulation:
        return self.simulation

    @sim.setter
    def sim(self, simulation: rebound.Simulation) -> None:
        self._sim_ = simulation

    @property
    def simulation(self) -> rebound.Simulation:
        """Create a simulation from the bodies and orbits of a system."""
        if sim := getattr(self, "_sim_", None):
            return sim
        sim = rebound.Simulation()
        sim.units = ("yr", "AU", "kg")
        _, length, time = self.base_units(sim)
        sim.ri_ias15.min_dt = (1 * u.s).to(time).value

        bodies = self.system.flattened
        for body in bodies:
            if orbit := body.orbit:
                if isinstance(body.parent, Body):
                    parent = bodies.index(body.parent)
                    sim.add(
                        hash=body.name,
                        primary=sim.particles[parent],
                        m=body.mass.to(u.kg).value,
                        a=(orbit.a.to(u.m) / const.au).value,
                        e=orbit.e.value,
                        inc=orbit.i.to(u.rad).value,
                        M=orbit.M.to(u.rad).value,
                        omega=orbit.omega.to(u.rad).value,
                        Omega=orbit.Omega.to(u.rad).value,
                    )
                else:
                    sim.add(
                        m=body.mass.to(u.kg).value,
                        a=(orbit.a.to(u.m) / const.au).value,
                        e=orbit.e.value,
                        inc=orbit.i.to(u.rad).value,
                        M=orbit.M.to(u.rad).value,
                        omega=orbit.omega.to(u.rad).value,
                        Omega=orbit.Omega.to(u.rad).value,
                    )
            else:
                sim.add(m=body.mass.to(u.kg).value)

        sim.move_to_com()

        rebx = reboundx.Extras(sim)
        gr = rebx.load_force("gr")
        rebx.add_force(gr)
        gr.params["c"] = const.c.to(length / time).value
        self._sim_ = sim
        return sim

    @simulation.setter
    def simulation(self, sim: rebound.Simulation) -> None:
        self._sim_ = sim

    def update_system(self, sim: rebound.Simulation) -> None:
        """Update the bodies and orbits of a system to match a simulation output."""
        (mass, length, time), angle = self.base_units(sim), u.rad
        for body, particle in zip(self.system.flattened, sim.particles):
            body.mass = (particle.m * mass).to(u.kg)
            body.position = (
                (particle.x * length).to(u.m),
                (particle.y * length).to(u.m),
                (particle.z * length).to(u.m),
            )
            if body.orbit:
                body.orbit.update(
                    a=particle.a * length,
                    e=particle.e * u.dimensionless_unscaled,
                    i=particle.inc * angle,
                    M=particle.M * angle,
                    omega=particle.omega * angle,
                    Omega=particle.Omega * angle,
                )

    def simulate(self, time: u.Quantity[u.s]) -> None:
        """propogate the system forward in time by a set time period."""
        sim = self.simulation
        _, _, timescale = self.base_units(sim)
        sim.integrate(time.to(timescale).value)
        self.update_system(sim)
        self.simulation = sim
