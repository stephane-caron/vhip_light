#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0
# Copyright 2016-2019 CNRS-UM LIRMM
# Copyright 2024 Inria

"""This example compares two balancers for the inverted pendulum model. The
first one (baseline) is based on proportional feedback of the 3D DCM
[Englsberger15]_. The second one (proposed) performs proportional feedback of a
4D DCM of the same model [Caron20]_.
"""

import IPython
import matplotlib
import matplotlib.pylab as plt
import numpy as np

from vhip_balancers import (
    Contact,
    InvertedPendulum,
    PolePlacementBalancer,
    VHIPQPBalancer,
    VRPBalancer,
)

MASS = 38.0  # [kg]

MAX_DCM_HEIGHT = 1.0  # [m]
MIN_DCM_HEIGHT = 0.5  # [m]

MAX_FORCE = 1000.0  # [N]
MIN_FORCE = 1.0  # [N]

K_P = 3.0  # proportional DCM feedback gain

assert K_P > 1.0, "DCM feedback gain needs to be greater than one"


class Pusher:
    """Send impulses to the inverted pendulum every once in a while.

    Args:
        pendulums: Inverted pendulums to de-stabilize.
        gain: Magnitude of velocity jumps.

    Notes:
        You know, I've seen a lot of people walkin' 'round // With tombstones
        in their eyes // But the pusher don't care // Ah, if you live or if you
        die
    """

    def __init__(self, pendulums, gain=0.1):
        super(Pusher, self).__init__()
        self.gain = gain
        self.nb_ticks = 0
        self.pendulums = pendulums
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def on_tick(self, dt):
        """Apply regular impulses to the inverted pendulum.

        Args:
            dt: Time step.
        """
        self.nb_ticks += 1
        one_sec = int(1.0 / dt)
        if self.started and self.nb_ticks % one_sec == 0:
            self.push()

    def push(self, dv=None, gain=None):
        if gain is None:
            gain = self.gain
        if dv is None:
            dv = 2.0 * np.random.random(3) - 1.0
            dv *= gain / np.linalg.norm(dv)
            print("Pusher: dv = {}".format(repr(dv)))
        for pendulum in self.pendulums:
            comd = pendulum.com.pd
            pendulum.com.set_vel(comd + dv)


class Plotter:

    def __init__(self, balancers):
        super(Plotter, self).__init__()
        self.plots = {
            "omega": [[] for stab in balancers],
            "xi_x": [[] for stab in balancers],
            "xi_y": [[] for stab in balancers],
            "xi_z": [[] for stab in balancers],
        }
        self.balancers = balancers

    def on_tick(self, sim):
        for i, stab in enumerate(self.balancers):
            cop = stab.pendulum.cop
            dcm = stab.dcm
            omega2 = stab.omega**2
            lambda_ = stab.pendulum.lambda_
            self.plots["xi_x"][i].append([dcm[0], cop[0]])
            self.plots["xi_y"][i].append([dcm[1], cop[1]])
            self.plots["xi_z"][i].append([dcm[2]])
            self.plots["omega"][i].append([omega2, lambda_])

    def plot(self, size=1000):
        matplotlib.rcParams["font.size"] = 14
        legends = {
            "omega": ("$\\omega^2$", "$\\lambda$"),
            "xi_x": ("$\\xi_x$", "$z_x$"),
            "xi_y": ("$\\xi_y$", "$z_y$"),
            "xi_z": ("$\\xi_z$",),
        }
        plt.clf()
        linestyles = ["-", ":", "--"]
        colors = ["b", "g", "r"]
        ref_omega = vrp_balancer.ref_omega
        ref_lambda = vrp_balancer.ref_lambda
        ref_dcm_p = vrp_balancer.ref_dcm
        refs = {
            "omega": [ref_omega**2, ref_lambda],
            "xi_x": [ref_dcm_p[0]],
            "xi_y": [ref_dcm_p[1]],
            "xi_z": [ref_dcm_p[2]],
        }
        for figid, figname in enumerate(self.plots):
            plt.subplot(411 + figid)
            for i, stab in enumerate(self.balancers):
                curves = zip(*self.plots[figname][i][-size:])
                trange = [dt * k for k in range(len(curves[0]))]
                for j, curve in enumerate(curves):
                    plt.plot(
                        trange, curve, linestyle=linestyles[i], color=colors[j]
                    )
            for ref in refs[figname]:
                plt.plot([trange[0], trange[-1]], [ref, ref], "k--")
            if figname == "xi_x":
                r_x_max = contact.p[0] + contact.shape[0]
                r_x_min = contact.p[0] - contact.shape[0]
                plt.plot([trange[0], trange[-1]], [r_x_max] * 2, "m:", lw=2)
                plt.plot([trange[0], trange[-1]], [r_x_min] * 2, "m:", lw=2)
                plt.ylim(r_x_min - 0.02, r_x_max + 0.02)
            if figname == "xi_y":
                r_y_max = contact.p[1] + contact.shape[1]
                r_y_min = contact.p[1] - contact.shape[1]
                plt.plot([trange[0], trange[-1]], [r_y_max] * 2, "m:", lw=2)
                plt.plot([trange[0], trange[-1]], [r_y_min] * 2, "m:", lw=2)
                plt.ylim(r_y_min - 0.01, r_y_max + 0.01)
            plt.legend(legends[figname])
            plt.grid(True)


def step(n_steps: int) -> None:
    for process in sim:
        process.step(dt)


def push_three_times():
    """Apply three pushes of increasing magnitude to the CoM.

    Note:
    ----
    This is the function used to generate Fig. 1 in the manuscript
    <https://hal.archives-ouvertes.fr/hal-02289919v1/document>.
    """
    step(10)
    pusher.push([0.0, 0.08, 0.0])
    step(40)
    pusher.push([0.0, 0.12, 0.0])
    step(50)
    pusher.push([0.0, 0.18, 0.0])
    step(100)


if __name__ == "__main__":
    dt = 0.03  # [s]

    contact = Contact((0.1, 0.05), pos=[0.0, 0.0, 0.0])
    init_pos = np.array([0.0, 0.0, 0.8])
    init_vel = np.zeros(3)
    pendulums = []
    balancers = []
    kp = 3.0

    pendulums.append(InvertedPendulum(init_pos, init_vel, contact))
    vhip_balancer = VHIPQPBalancer(pendulums[-1])
    balancers.append(vhip_balancer)

    pendulums.append(InvertedPendulum(init_pos, init_vel, contact))
    vrp_balancer = VRPBalancer(pendulums[-1])
    balancers.append(vrp_balancer)

    pendulums.append(InvertedPendulum(init_pos, init_vel, contact))
    bonus_balancer = PolePlacementBalancer(pendulums[-1], k_z=100)
    balancers.append(bonus_balancer)

    pusher = Pusher(pendulums)
    plotter = Plotter(balancers)

    sim = []
    for balancer, pendulum in zip(balancers, pendulums):
        sim.append(balancer)  # before pendulum
        sim.append(pendulum)
    sim.append(plotter)  # before pusher
    sim.append(pusher)

    def reset():
        for stab in balancers:
            stab.reset_pendulum()
        sim.step()

    sim.step(42)  # go to reference
    impulse = np.array([0.0, -0.09, 0.0])
    # push_three_times()  # scenario for Fig. 1 of the paper
    reset()

    if IPython.get_ipython() is None:  # give the user a prompt
        IPython.embed()
    IPython.get_ipython().magic("pylab")  # for plots
