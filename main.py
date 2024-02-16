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

from typing import List

import IPython
import numpy as np

from vhip_balancers import (
    Contact,
    InvertedPendulum,
    Plotter,
    Process,
    Pusher,
    VHIPQPBalancer,
    VRPBalancer,
)

FEEDBACK_GAIN = 3.0  # proportional DCM feedback gain
MASS = 38.0  # [kg]
MAX_DCM_HEIGHT = 1.0  # [m]
MAX_FORCE = 1000.0  # [N]
MIN_DCM_HEIGHT = 0.5  # [m]
MIN_FORCE = 1.0  # [N]

assert FEEDBACK_GAIN > 1.0, "DCM feedback gain needs to be greater than one"


if __name__ == "__main__":
    dt = 0.03  # [s]

    contact = Contact(shape=(0.1, 0.05), pos=[0.0, 0.0, 0.0])
    init_pos = np.array([0.0, 0.0, 0.8])
    init_vel = np.zeros(3)
    pendulums = []
    balancers = []
    kp = 3.0

    pendulums.append(InvertedPendulum(init_pos, init_vel, contact, mass=MASS))
    vhip_balancer = VHIPQPBalancer(
        pendulums[-1],
        kp=FEEDBACK_GAIN,
        max_force=MAX_FORCE,
        min_force=MIN_FORCE,
        max_dcm_height=MAX_DCM_HEIGHT,
        min_dcm_height=MIN_DCM_HEIGHT,
    )
    balancers.append(vhip_balancer)

    pendulums.append(InvertedPendulum(init_pos, init_vel, contact, mass=MASS))
    vrp_balancer = VRPBalancer(pendulums[-1], kp=FEEDBACK_GAIN)
    balancers.append(vrp_balancer)

    pusher = Pusher(pendulums)
    plotter = Plotter(
        balancers,
        ref_omega=vrp_balancer.ref_omega,
        ref_lambda=vrp_balancer.ref_lambda,
        ref_dcm=vrp_balancer.ref_dcm,
    )

    processes: List[Process] = []
    for balancer, pendulum in zip(balancers, pendulums):
        processes.append(balancer)  # before pendulum
        processes.append(pendulum)
    processes.append(plotter)  # before pusher
    processes.append(pusher)

    def step(nb_steps: int = 1) -> None:
        for step in range(nb_steps):
            for process in processes:
                process.step(dt)

    def push_three_times():
        """Apply three pushes of increasing magnitude to the CoM.

        Note:
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

    def reset():
        for stab in balancers:
            stab.reset_pendulum()
        step()

    step(42)  # go to reference
    impulse = np.array([0.0, -0.09, 0.0])
    push_three_times()  # scenario for Fig. 1 of the paper
    reset()

    if IPython.get_ipython() is None:  # give the user a prompt
        IPython.embed()
    IPython.get_ipython().magic("pylab")  # for plots
