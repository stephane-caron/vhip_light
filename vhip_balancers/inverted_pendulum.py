#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0
# Copyright 2024 Inria

from warnings import warn

import numpy as np

from .gravity import GRAVITY
from .point import Point


class InvertedPendulum:

    def __init__(
        self,
        pos,
        vel,
        contact,
        mass: float,
        lambda_min: float = 1e-5,
        lambda_max: float = None,
        clamp=True,
    ):
        super(InvertedPendulum, self).__init__()
        com = Point(pos, vel)
        self.clamp = clamp
        self.com = com
        self.contact = contact
        self.mass = mass
        self.cop = contact.p
        self.lambda_ = -GRAVITY[2] / (com.z - contact.z)
        self.lambda_max = lambda_max
        self.lambda_min = lambda_min

    def set_cop(self, cop, clamp=None):
        """Update the CoP location on the contact surface.

        Args:
            cop: New CoP location in the world frame.
            clamp: Clamp CoP within the contact area if it lies outside.
                Overrides ``self.clamp``.
        """
        if self.clamp if clamp is None else clamp:
            cop_local = np.dot(self.contact.R.T, cop - self.contact.p)
            if cop_local[0] >= self.contact.shape[0]:
                cop_local[0] = self.contact.shape[0] - 1e-5
            elif cop_local[0] <= -self.contact.shape[0]:
                cop_local[0] = -self.contact.shape[0] + 1e-5
            if cop_local[1] >= self.contact.shape[1]:
                cop_local[1] = self.contact.shape[1] - 1e-5
            elif cop_local[1] <= -self.contact.shape[1]:
                cop_local[1] = -self.contact.shape[1] + 1e-5
            cop = self.contact.p + np.dot(self.contact.R, cop_local)
        elif __debug__:
            cop_check = np.dot(self.contact.R.T, cop - self.contact.p)
            if abs(cop_check[0]) > 1.05 * self.contact.shape[0]:
                warn("CoP crosses contact area along sagittal axis")
            if abs(cop_check[1]) > 1.05 * self.contact.shape[1]:
                warn("CoP crosses contact area along lateral axis")
            if abs(cop_check[2]) > 0.01:
                warn("CoP does not lie on contact area")
        self.cop = cop

    def set_lambda(self, lambda_, clamp=None):
        """Update the leg stiffness coefficient.

        Parameters
        ----------
        lambda_ : scalar
            Leg stiffness coefficient (positive).
        clamp : bool, optional
            Clamp value if it exits the [lambda_min, lambda_max] interval.
            Overrides ``self.clamp``.
        """
        if self.clamp if clamp is None else clamp:
            if self.lambda_min is not None and lambda_ < self.lambda_min:
                lambda_ = self.lambda_min
            if self.lambda_max is not None and lambda_ > self.lambda_max:
                lambda_ = self.lambda_max
        elif __debug__:
            if self.lambda_min is not None and lambda_ < self.lambda_min:
                warn("Stiffness %f below %f" % (lambda_, self.lambda_min))
            if self.lambda_max is not None and lambda_ > self.lambda_max:
                warn("Stiffness %f above %f" % (lambda_, self.lambda_max))
        self.lambda_ = lambda_

    def integrate(self, duration):
        """Integrate dynamics forward for a given duration.

        Parameters
        ----------
        duration : scalar
            Duration of forward integration.
        """
        omega = np.sqrt(self.lambda_)
        p0 = self.com.p
        pd0 = self.com.pd
        ch, sh = np.cosh(omega * duration), np.sinh(omega * duration)
        vrp = self.cop - GRAVITY / self.lambda_
        p = p0 * ch + pd0 * sh / omega - vrp * (ch - 1.0)
        pd = pd0 * ch + omega * (p0 - vrp) * sh
        self.com.set_pos(p)
        self.com.set_vel(pd)

    def step(self, dt: float):
        self.integrate(dt)
