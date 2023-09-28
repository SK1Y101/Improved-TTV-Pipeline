import logging

import astropy.units as u

from planetary_system import system
from simulation import Simulation
from visualiser import SystemVisualiser

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)-5.5s]  %(message)s"
)

if __name__ == "__main__":
    sol = system.presets.innerPlanets
    plot = SystemVisualiser(sol)
    sim = Simulation(sol)

    sol.log_bodies()

    sol.plot_system()

    sol.simulate(20 * u.day)

    sol.plot_system()
