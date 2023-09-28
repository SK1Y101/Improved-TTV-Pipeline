import astropy.units as u

from planetary_system import bodies, orbits

pbodies = bodies.presets
porbits = orbits.presets


class TestOrbitConversion:
    def test_cartesian_kepler(self) -> None:
        sun = pbodies.Sun
        earth = pbodies.Earth
        earth.orbit = porbits.Earth
        sun.add_body(earth)

        assert earth.orbit.a == 149598023 * u.km

        # Test the conversions work both ways
        objs = [earth.orbit, earth.orbit.kepler.cartesian, earth.orbit.cartesian.kepler]

        for primary in objs:
            for secondary in objs:
                assert primary.line_element == secondary.line_element

    def test_TLE(self) -> None:
        earth_orbit = porbits.Earth

        assert len(earth_orbit.line_element) == len(
            "0000000+00 000.0000 000.0000 0000000 000.0000 000.0000"
        )

        tle_float = [
            float(elem)
            if "." in elem
            else float(f"0.{elem}".replace("+", "e+").replace("-", "e-"))
            for elem in earth_orbit.line_element.split()
        ]
        assert tle_float == [
            149598000000.0,
            0.0001,
            348.7394,
            0.0167086,
            114.2078,
            358.617,
        ]
