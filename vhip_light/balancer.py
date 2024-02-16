#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0
# Copyright 2024 Inria

import numpy as np

from .gravity import GRAVITY
from .process import Process


class Balancer(Process):
    """Base class for balancers.

    Attributes:
        contact: Contact frame and area dimensions.
        dcm: Position of the DCM in the world frame.
        omega: Instantaneous natural frequency of the pendulum.
        pendulum: Measured state of the reduced model.
        ref_com: Desired center of mass (CoM) position.
        ref_comd: Desired CoM velocity.
        ref_cop: Desired center of pressure (CoP).
        ref_lambda: Desired normalized leg stiffness.
        ref_omega: Desired natural frequency.
    """

    def __init__(self, pendulum, ref_offset=None):
        super().__init__()
        if ref_offset is None:
            ref_offset = np.array([0.0, 0.0, 0.0])  # [m]
        ref_com = pendulum.com.p + ref_offset
        n = pendulum.contact.normal
        lambda_ = -np.dot(n, GRAVITY) / np.dot(n, ref_com - pendulum.contact.p)
        omega = np.sqrt(lambda_)
        ref_cop = ref_com + GRAVITY / lambda_
        assert abs(lambda_ - pendulum.lambda_) < 1e-5
        self.contact = pendulum.contact
        self.dcm = ref_com
        self.omega = omega
        self.pendulum = pendulum
        self.ref_com = ref_com
        self.ref_comd = np.zeros(3)
        self.ref_cop = ref_cop
        self.ref_lambda = lambda_
        self.ref_omega = omega

    def reset_pendulum(self):
        """Reset inverted pendulum to its reference state."""
        self.omega = self.ref_omega
        self.pendulum.com.set_pos(self.ref_com)
        self.pendulum.com.set_vel(self.ref_comd)
        self.pendulum.set_cop(self.ref_cop)
        self.pendulum.set_lambda(self.ref_lambda)

    def step(self, dt: float) -> None:
        """Set inverted pendulum CoP and stiffness inputs."""
        Delta_r, Delta_lambda = self.compute_compensation(dt)
        cop = self.ref_cop + np.dot(self.contact.R[:3, :2], Delta_r)
        lambda_ = self.ref_lambda + Delta_lambda
        self.pendulum.set_cop(cop)
        self.pendulum.set_lambda(lambda_)
