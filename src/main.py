import logging

from planetary_system import system
from visualiser import SystemVisualiser

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)-5.5s]  %(message)s"
)

if __name__ == "__main__":
    sol = system.presets.solarSystem

    sol.log_bodies()

    sol.logger.info(sol.centre_of_mass)

    # sol.plot_system(style="simple")
