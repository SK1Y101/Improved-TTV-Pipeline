import astropy.constants as const
import astropy.units as u
import numpy as np

from planetary_system import bodies

pbodies = bodies.presets


class TestSystems:
    def test_defined_body_properties(self) -> None:
        assert round(pbodies.Sun.temperature.value) == 5772
        assert round(max(pbodies.Earth.radius).value) == round(
            const.R_earth.to(u.km).value
        )

    def test_derived_properties(self) -> None:
        properties = {"name": "test", "mass": 1e24 * u.kg, "radius": 7e4 * u.km}
        planet = bodies.planet(
            name=properties["name"],
            mass=properties["mass"],
            radius=properties["radius"],
        )
        assert planet.radius == properties["radius"]
        assert planet.density == properties["mass"] * 3 / (
            4 * properties["radius"] ** 3 * np.pi
        )
