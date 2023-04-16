import astropy.constants as const
import astropy.units as u
import numpy as np


def cosineramp(start_val, end_val, length):
    t = 0.5 * (1 + np.cos(np.linspace(0, np.pi, length)))
    return t * (start_val - end_val) + end_val


# create a ramped curve
def rampupdown(fulltime, ramptime, maxval, minval):
    fulltime, ramptime = round(fulltime), round(ramptime)
    rampup = cosineramp(maxval, minval, ramptime)
    rampdown = cosineramp(minval, maxval, ramptime)
    return np.concatenate([rampup, minval * np.ones(fulltime - ramptime * 2), rampdown])


def create_lc(star, planet, obs_sep=1 * u.hour):
    obs_step = obs_sep.to(u.second)

    # compute som observables
    def angrad(srad, prad, sdist, pdist):
        s_ang_rad = np.arctan2(srad, sdist)
        p_ang_rad = np.arctan2(prad, pdist)
        return s_ang_rad, p_ang_rad

    s_ang_rad, p_ang_rad = angrad(
        star["radius"],
        planet["radius"],
        star["distance"],
        (star["distance"] - planet["sma"]),
    )
    mu = const.G * (star["mass"] + planet["mass"])
    p_orb_vel = np.sqrt(mu / planet["sma"]).to(u.m / u.s)
    i = planet["inclination"] if "inclination" in planet else 0
    b = (planet["sma"] * np.sin(i.to(u.rad)) / s_ang_rad).value

    print(
        "maximum allowable impact parameter = {} ".format(
            (s_ang_rad - p_ang_rad) ** 2 / s_ang_rad**2
        )
    )
    print("This impact parameter = {}".format(b))

    # transit parameters
    t_depth = 1 - min(1, p_ang_rad**2 / s_ang_rad**2)
    # angular radius of the deep transit
    t_ang_rad = 2 * np.sqrt((s_ang_rad - p_ang_rad) ** 2 - (b * s_ang_rad) ** 2)
    t_width = star["distance"] * 2 * np.tan(0.5 * t_ang_rad)
    t_time = (t_width / p_orb_vel).to(u.s)

    t_full_ang_rad = 2 * np.sqrt((s_ang_rad + p_ang_rad) ** 2 - (b * s_ang_rad) ** 2)
    t_full_width = star["distance"] * 2 * np.tan(0.5 * t_full_ang_rad)
    t_full_time = (t_full_width / p_orb_vel).to(u.s)

    # ramp parameters
    t_full_idxs = (t_full_time / obs_step).value
    t_ramp_idxs = (0.5 * (t_full_time - t_time) / obs_step).value

    # create the transit curve

    t_curve = rampupdown(t_full_idxs, t_ramp_idxs, 1, t_depth)

    return t_curve, np.linspace(0, t_full_time, round(t_full_idxs)) - t_full_time * 0.5
