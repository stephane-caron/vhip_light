#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0
# Copyright 2024 Inria

import cvxpy
import numpy as np

from .balancer import Balancer
from .gravity import GRAVITY
from .inverted_pendulum import InvertedPendulum


class VHIPBalancer(Balancer):
    """Proportional feedback of the 4D DCM of the VHIP.

    Args:
        pendulum: Inverted pendulum to stabilize.

    Notes:
        This implementation uses CVXPY <https://www.cvxpy.org/>. Using this
        modeling language here allowed us to try various formulations of the
        controller before converging on this one. We can only praise the
        agility of this approach, as opposed to e.g. writing QP matrices
        directly.

        See "Biped Stabilization by Linear Feedback of the Variable-Height
        Inverted Pendulum Model" (Caron, 2019) for detail on the controller
        itself.
    """

    def __init__(
        self,
        pendulum: InvertedPendulum,
        kp: float,
        max_dcm_height: float,
        max_force: float,
        min_dcm_height: float,
        min_force: float,
    ):
        super(VHIPBalancer, self).__init__(pendulum)
        r_d_contact = np.dot(self.contact.R.T, self.ref_cop - self.contact.p)[
            :2
        ]
        self.kp = kp
        self.max_dcm_height = max_dcm_height
        self.max_force = max_force
        self.min_dcm_height = min_dcm_height
        self.min_force = min_force
        self.r_contact_max = np.array(self.contact.shape)
        self.ref_cop_contact = r_d_contact
        self.ref_dcm = self.ref_com
        self.ref_vrp = self.ref_com

    def compute_compensation(self, dt: float):
        """Compute CoP and normalized leg stiffness compensation."""
        # Measurements
        Delta_com = self.pendulum.com.p - self.ref_com
        Delta_comd = self.pendulum.com.pd - self.ref_comd
        height = np.dot(
            self.contact.normal, self.pendulum.com.p - self.contact.p
        )
        lambda_d = self.ref_lambda
        measured_comd = self.pendulum.com.pd
        nu_d = self.ref_vrp
        omega_d = self.ref_omega
        r_d_contact = self.ref_cop_contact
        xi_d = self.ref_dcm

        # Force limits
        lambda_max = self.max_force / (self.pendulum.mass * height)
        lambda_min = self.min_force / (self.pendulum.mass * height)
        omega_max = np.sqrt(lambda_max)
        omega_min = np.sqrt(lambda_min)

        # Optimization variables
        Delta_lambda = cvxpy.Variable(1)
        Delta_nu = cvxpy.Variable(3)
        Delta_omega = cvxpy.Variable(1)
        Delta_r = cvxpy.Variable(2)
        u = cvxpy.Variable(3)

        # Linearized variation dynamics
        Delta_xi = (
            Delta_com
            + Delta_comd / omega_d
            - measured_comd / (omega_d**2) * Delta_omega
        )
        Delta_omegad = 2 * omega_d * Delta_omega - Delta_lambda
        Delta_r_world = self.contact.R[:3, :2] * Delta_r
        r_contact = r_d_contact + Delta_r
        lambda_ = lambda_d + Delta_lambda
        omega = omega_d + Delta_omega

        # Pole placement
        Delta_xid = (
            Delta_lambda * (xi_d - nu_d)
            + lambda_d * (Delta_xi - Delta_nu)
            + -Delta_omega * lambda_d * (xi_d - nu_d) / omega_d
        ) / omega_d

        # Kinematic DCM height constraint
        xi_z = self.ref_dcm[2] + Delta_xi[2] + 1.5 * dt * Delta_xid[2]

        # Cost function
        costs = []
        sq_costs = [(1.0, u[0]), (1.0, u[1]), (1e-3, u[2])]
        for weight, expr in sq_costs:
            costs.append((weight, cvxpy.sum_squares(expr)))
        cost = sum(weight * expr for (weight, expr) in costs)

        # Quadratic program
        prob = cvxpy.Problem(
            objective=cvxpy.Minimize(cost),
            constraints=[
                Delta_xid
                == lambda_d / omega_d * ((1 - self.kp) * Delta_xi + u),
                Delta_omegad == omega_d * (1 - self.kp) * Delta_omega,
                Delta_nu
                == Delta_r_world + GRAVITY * Delta_lambda / lambda_d**2,
                cvxpy.abs(r_contact) <= self.r_contact_max,
                lambda_ <= lambda_max,
                lambda_ >= lambda_min,
                xi_z <= self.max_dcm_height,
                xi_z >= self.min_dcm_height,
                omega <= omega_max,
                omega >= omega_min,
            ],
        )
        prob.solve()

        # Read outputs from solution
        Delta_lambda_opt = Delta_lambda.value
        Delta_r_opt = np.array(Delta_r.value).reshape((2,))
        self.omega = omega_d + Delta_omega.value
        self.dcm = self.pendulum.com.p + self.pendulum.com.pd / self.omega
        return (Delta_r_opt, Delta_lambda_opt)
