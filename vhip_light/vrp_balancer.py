#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0
# Copyright 2024 Inria

import numpy as np

from .balancer import Balancer
from .gravity import GRAVITY


class VRPBalancer(Balancer):
    """Inverted pendulum balancerbased on proportional feedback of the
    3D divergent component of motion (DCM) applied to the virtual repellent
    point (VRP).

    Attributes:
        kp: Proportional gain.
        ref_dcm: Desired (3D) divergent component of motion.
        ref_vrp: Desired virtual repellent point (VRP).

    Notes:
        See "Three-Dimensional Bipedal Walking Control Based on Divergent
        Component of Motion" (Englsberger et al., IEEE Transactions on
        Robotics) for details.
    """

    def __init__(self, pendulum, kp: float):
        """Initialize balancer.

        Args:
            pendulum: Inverted pendulum to stabilize.
            kp: Proportional gain.
        """
        super(VRPBalancer, self).__init__(pendulum)
        self.ref_dcm = self.ref_com
        self.ref_vrp = self.ref_com
        self.kp = kp

    def compute_compensation(self, dt: float):
        """Compute CoP and normalized leg stiffness compensation."""
        omega = self.omega
        com = self.pendulum.com.p
        comd = self.pendulum.com.pd
        dcm = com + comd / omega
        Delta_dcm = dcm - self.ref_dcm
        vrp = self.ref_vrp + self.kp * Delta_dcm
        n = self.pendulum.contact.n
        gravito_inertial_force = omega**2 * (com - vrp) - GRAVITY
        displacement = com - self.pendulum.contact.p
        lambda_ = np.dot(n, gravito_inertial_force) / np.dot(n, displacement)
        cop = com - gravito_inertial_force / lambda_
        Delta_r = np.dot(self.contact.R.T, cop - self.ref_cop)[:2]
        Delta_lambda = lambda_ - self.ref_lambda
        self.dcm = dcm
        return (Delta_r, Delta_lambda)
