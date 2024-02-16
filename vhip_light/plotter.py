#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0
# Copyright 2024 Inria

import matplotlib
import matplotlib.pylab as plt
import numpy as np
from numpy.typing import NDArray

from .contact import Contact
from .process import Process


class Plotter(Process):

    def __init__(
        self,
        balancers,
        contact: Contact,
        ref_omega: float,
        ref_lambda: float,
        ref_dcm: NDArray[float],
    ):
        super(Plotter, self).__init__()
        self.plots = {
            "omega": [[] for stab in balancers],
            "xi_x": [[] for stab in balancers],
            "xi_y": [[] for stab in balancers],
            "xi_z": [[] for stab in balancers],
        }
        self.balancers = balancers
        self.contact = contact
        self.ref_omega = ref_omega
        self.ref_lambda = ref_lambda
        self.ref_dcm_p = ref_dcm

    def plot(self, dt: float, size=1000):
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
        refs = {
            "omega": [self.ref_omega**2, self.ref_lambda],
            "xi_x": [self.ref_dcm_p[0]],
            "xi_y": [self.ref_dcm_p[1]],
            "xi_z": [self.ref_dcm_p[2]],
        }
        for figid, figname in enumerate(self.plots):
            plt.subplot(411 + figid)
            for i, stab in enumerate(self.balancers):
                curves = np.array(list(zip(*self.plots[figname][i][-size:])))
                trange = [dt * k for k in range(len(curves[0]))]
                for j, curve in enumerate(curves):
                    plt.plot(
                        trange, curve, linestyle=linestyles[i], color=colors[j]
                    )
            for ref in refs[figname]:
                plt.plot([trange[0], trange[-1]], [ref, ref], "k--")
            if figname == "xi_x":
                r_x_max = self.contact.p[0] + self.contact.shape[0]
                r_x_min = self.contact.p[0] - self.contact.shape[0]
                plt.plot([trange[0], trange[-1]], [r_x_max] * 2, "m:", lw=2)
                plt.plot([trange[0], trange[-1]], [r_x_min] * 2, "m:", lw=2)
                plt.ylim(r_x_min - 0.02, r_x_max + 0.02)
            if figname == "xi_y":
                r_y_max = self.contact.p[1] + self.contact.shape[1]
                r_y_min = self.contact.p[1] - self.contact.shape[1]
                plt.plot([trange[0], trange[-1]], [r_y_max] * 2, "m:", lw=2)
                plt.plot([trange[0], trange[-1]], [r_y_min] * 2, "m:", lw=2)
                plt.ylim(r_y_min - 0.01, r_y_max + 0.01)
            plt.legend(legends[figname], loc="upper right")
            plt.grid(True)

    def step(self, dt: float) -> None:
        for i, stab in enumerate(self.balancers):
            cop = stab.pendulum.cop
            dcm = stab.dcm
            omega2 = stab.omega**2
            lambda_ = stab.pendulum.lambda_
            self.plots["xi_x"][i].append([dcm[0], cop[0]])
            self.plots["xi_y"][i].append([dcm[1], cop[1]])
            self.plots["xi_z"][i].append([dcm[2]])
            self.plots["omega"][i].append([omega2, lambda_])
