import astropy.constants as const
import astropy.units as u
import matplotlib.gridspec as gridspec  # Plotting
import numpy as np
from matplotlib import pylab as plt
from matplotlib.animation import FuncAnimation

# import modules
from matplotlib.collections import LineCollection

from lightcurve import *


def draw_orbit(ax, rad=1, theta=0, col=(0, 0, 0), points=360):
    if max(col) > 1:
        col = np.array(col) / 255
    # create the rotation
    psi = np.linspace(0, 1.5 * np.pi, points) + theta
    # fetch the circle coordinates
    cx, cy = rad * np.sin(psi), rad * np.cos(psi)
    # make the orbit
    orbit = np.zeros((points - 1, 2, 2))
    orbit[:, 0, 0] = cx[:-1]
    orbit[:, 0, 1] = cy[:-1]
    orbit[:, 1, 0] = cx[1:]
    orbit[:, 1, 1] = cy[1:]
    # ceate the colour as rgba
    colour = np.zeros((points, 4))
    # set rgb
    colour[:, 0:3] = col
    # fade a
    colour[:, 3] = np.linspace(0, 1, points)[::-1]
    # and plot
    ax.add_collection(LineCollection(orbit, color=colour, lw=1))


def draw_body(ax, sma, rad, theta=0, col=(0, 0, 0)):
    if max(col) > 1:
        col = np.array(col) / 255
    ax.add_artist(
        plt.Circle((sma * np.sin(theta), sma * np.cos(theta)), radius=rad, color=col)
    )


def draw_orbiting_body(ax, sma, rad, theta=0, col=(0, 0, 0)):
    if max(col) > 1:
        col = np.array(col) / 255
    # draw orbit
    draw_orbit(ax, sma, theta, col)
    # draw body
    draw_body(ax, sma, rad, theta, col)


star = {
    "distance": 50 * u.lyr,
    "radius": 696342 * u.km,
    "mass": 1.98855 * 10**30 * u.kg,
    "lum_fluc": 1e-3 * np.sqrt(6 / 9),
}
planet = {
    "sma": 1 * u.AU,
    "radius": 3 * 69911 * u.km,
    "mass": 1.8982 * 10**27 * u.kg,
    "inclination": 0 * u.degree,
}
p3 = (planet["sma"] / u.AU) ** (2 / 3)
a3 = p3 ** (3 / 2)

lc, time = create_lc(star, planet, obs_sep=90 * u.second)
# make the base bucket shaped
lc += (time.to(u.hour).value / (50)) ** 2
lc *= 1 / max(lc)
# add ones either side
lc_ = np.ones(4 * int(len(lc)))
lc = np.concatenate([lc_, lc, lc_]) ** 8
full_time = (time[1] - time[0]) * len(lc)
time = np.linspace(0, full_time, len(lc)) - full_time * 0.5
time = time.to(u.hour)
# add a slight curve due to reflected planet light
lc += (time.value / (250)) ** 2
# conversion functions
a = lambda p: p ** (3 / 2)
posx = lambda m, p, frame: m * a(p) * np.sin(frame / p)
posy = lambda m, p, frame: m * a(p) * np.cos(frame / p)
radscale = lambda r: r ** (1 / 3) * 0.05

fig, ax = plt.subplots(1, 1)  # , figsize=(4, 4))
outer = 2
m1, m2, m3, m4, m5 = 1.5, 1, 0.5, 0.5, 0.25
resonance = [3, 4, 5, 6, 7]
p1, p2, p3, p4, p5 = [float(p3) * res / resonance[3] for res in resonance]
lim = max(a(p1), a(p2), a(p3), a(p4), a(p5)) + 0.51


def lcm(*ints):
    import math

    minpow = 10 ** abs(math.floor(math.log10(min(ints))))
    return math.lcm(*[int(a * minpow) for a in ints]) / minpow


def offset(frame, period, angle):
    return frame + 2 * np.pi * angle * period


def update(frame, ax=ax):
    # clear frame
    ax.clear()
    # planets
    lcoff = a(p3) + 1.2
    t3 = offset(frame, p3, 5 / 8)
    t1 = offset(frame, p1, 1 / 3)
    t2 = offset(frame, p2, -1 / 3)
    t4 = offset(frame, p4, 3 / 7)
    t5 = offset(frame, p5, 6 / 8.5)
    draw_orbiting_body(ax, a(p1), radscale(m1), t1 / p1 - t3 / p3, (136, 204, 238))
    draw_orbiting_body(ax, a(p2), radscale(m2), t2 / p2 - t3 / p3, (68, 170, 153))
    draw_orbiting_body(ax, a(p3), radscale(m3), np.pi, (51, 34, 136))
    draw_orbiting_body(ax, a(p4), radscale(m4), t4 / p4 - t3 / p3, (254, 200, 216))
    draw_orbiting_body(ax, a(p5), radscale(m5), t5 / p5 - t3 / p3, (255, 223, 211))
    # compute centre of mass of planets
    brx = (
        posx(m1, p1, t1 - t3 * p1 / p3)
        + posx(m2, p2, t2 - t3 * p2 / p3)
        + posx(m3, p3, np.pi)
        + posx(m4, p4, t4 - t3 * p4 / p3)
        + posx(m5, p5, t5 - t3 * p5 / p3)
    )
    bry = (
        posy(m1, p1, t1 - t3 * p1 / p3)
        + posy(m2, p2, t2 - t3 * p2 / p3)
        + posy(m3, p3, np.pi)
        + posy(m4, p4, t4 - t3 * p4 / p3)
        + posy(m5, p5, t5 - t3 * p5 / p3)
    )
    # star is opposite the barycentre
    ms = 10
    sx, sy = -brx / ms, -bry / ms
    sa, st = (sx**2 + sy**2) ** 0.5, 0.5 * np.pi - np.arctan2(sy, sx)
    draw_body(ax, sa, radscale(ms), st, (221, 204, 119))
    # barycentre
    ax.add_artist(plt.Circle((0, 0), radius=0.01, color="black"))
    # ligcurves
    ax.plot(0.05 * time.value, lc - lcoff, label="Unperturbed Lightcurve")
    ax.plot(0.05 * time.value + sx, lc - lcoff, label="Perturbed Lightcurve")
    # limits
    ax.set_xlim((-lim, lim))
    ax.set_ylim((-(lim + 0.5), lim - 0.5))
    # ax.set_facecolor("lightgray")
    # ax.axis("off")
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)
    ax.set_aspect("equal")
    fig.tight_layout()
    # ax.legend()


# side by side comparison eh?
# w, h = 2, 2
# fig = plt.figure(figsize=(10*w,110*h))
# gs = gridspec.GridSpec(1, 3, wspace=0)
# axes = [plt.subplot(gs[x]) for x in range(3)]

# update(np.pi, axes[0])
# axes[0].set_title("Early transit", fontsize="xx-large")
# axes[0].set_xlim([-a3*outer-0.01, a3*outer+0.01])
# update(0, axes[1])
# axes[1].set_title("Expected transit", fontsize="xx-large")
# axes[1].set_xlim([-a3*outer-0.01, a3*outer+0.01])
# update(-np.pi, axes[2])
# axes[2].set_title("Late transit", fontsize="xx-large")
# axes[2].set_xlim([-a3*outer-0.01, a3*outer+0.01])
# plt.savefig("TTVDueToInnerBarycentre.pdf", bbox_inches="tight")
# plt.show()

scale = lcm(p1, p2, float(p3), p4)
print(scale)
ani = FuncAnimation(
    fig,
    update,
    frames=np.linspace(0, 2 * np.pi * scale, 360 * round(scale / 4))[::-1],
    interval=1,
    repeat=False,
)
plt.show()
ani.save(
    "TTVModelAnimation_Large.gif",
    writer="imagemagick",
    fps=60,
    progress_callback=lambda i, n: print(f"Saving frame {i} of {n}"),
    savefig_kwargs={"bbox_inches": "tight", "pad_inches": 0},
)
