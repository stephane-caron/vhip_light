#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0
# Copyright 2024 Inria

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from qpsolvers import SolverNotFound, available_solvers, solve_qp

from .inverted_pendulum import InvertedPendulum
from .vhip_balancer import VHIPBalancer


@dataclass
class VHIPQPResult:

    Delta_x: NDArray[float]

    @property
    def Delta_xi(self) -> NDArray[float]:
        return self.Delta_x[0:3]

    @property
    def Delta_omega(self) -> float:
        return self.Delta_x[3]

    @property
    def Delta_z(self) -> NDArray[float]:
        return self.Delta_x[4:6]

    @property
    def Delta_lambda(self) -> float:
        return self.Delta_x[6]

    @property
    def Delta_sigma(self) -> NDArray[float]:
        return self.Delta_x[7:10]


class VHIPQPBalancer(VHIPBalancer):
    """Proportional feedback of the 4D DCM of the VHIP.

    Args:
        pendulum: Inverted pendulum to stabilize.

    Notes:
        This implementation transcripts QP matrices from
        :class:`VHIPBalancer`. We checked that the two produce the same
        outputs before switching to C++ in
        <https://github.com/stephane-caron/vhip_walking_controller/>. (This
        step would not have been necessary if we had a modeling language for
        convex optimization directly in C++.)
    """

    def __init__(
        self,
        pendulum: InvertedPendulum,
        kp: float,
        max_dcm_height: float,
        max_force: float,
        min_dcm_height: float,
        min_force: float,
        solver: str = "quadprog",
    ):
        super().__init__(
            pendulum=pendulum,
            kp=kp,
            max_dcm_height=max_dcm_height,
            max_force=max_force,
            min_dcm_height=min_dcm_height,
            min_force=min_force,
        )
        if solver not in available_solvers:
            raise SolverNotFound(
                f"solver '{solver}' is not in {available_solvers=}"
            )
        self.solver = solver

    def compute_compensation(self, dt: float):
        """Compute CoP and normalized leg stiffness compensation."""
        Delta_com = self.pendulum.com.p - self.ref_com
        Delta_comd = self.pendulum.com.pd - self.ref_comd
        measured_comd = self.pendulum.com.pd
        lambda_d = self.ref_lambda
        nu_d = self.ref_vrp
        omega_d = self.ref_omega
        r_d = self.ref_cop
        r_d_contact = self.ref_cop_contact
        xi_d = self.ref_dcm
        height = np.dot(
            self.contact.normal, self.pendulum.com.p - self.contact.p
        )
        lambda_max = self.max_force / (self.pendulum.mass * height)
        lambda_min = self.min_force / (self.pendulum.mass * height)
        omega_max = np.sqrt(lambda_max)
        omega_min = np.sqrt(lambda_min)

        A = np.vstack(
            [
                np.hstack(
                    [
                        -self.kp * np.eye(3),
                        (xi_d - nu_d).reshape((3, 1)) / omega_d,
                        self.contact.R[:3, :2],
                        (r_d - xi_d).reshape((3, 1)) / lambda_d,
                        np.eye(3),
                    ]
                ),
                np.hstack(
                    [
                        np.eye(3),
                        measured_comd.reshape((3, 1)) / omega_d**2,
                        np.zeros((3, 2)),
                        np.zeros((3, 1)),
                        np.zeros((3, 3)),
                    ]
                ),
                np.hstack(
                    [
                        np.zeros((1, 3)),
                        omega_d * (1 + self.kp) * np.eye(1),
                        np.zeros((1, 2)),
                        -1 * np.eye(1),
                        np.zeros((1, 3)),
                    ]
                ),
            ]
        )
        b = np.hstack(
            [
                np.zeros(3),
                Delta_com + Delta_comd / omega_d,
                np.zeros(1),
            ]
        )

        G_cop = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, +1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, +1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        h_cop = np.array(
            [
                self.contact.shape[0] - r_d_contact[0],
                self.contact.shape[0] + r_d_contact[0],
                self.contact.shape[1] - r_d_contact[1],
                self.contact.shape[1] + r_d_contact[1],
            ]
        )

        G_lambda = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, +1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
            ]
        )
        h_lambda = np.array([lambda_max - lambda_d, lambda_d - lambda_min])

        G_omega = np.array(
            [
                [0.0, 0.0, 0.0, +1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        h_omega = np.array([omega_max - omega_d, omega_d - omega_min])

        g_sigma = 1.5 * lambda_d * dt / omega_d
        g_xi = 1 + g_sigma * (1 - self.kp)
        G_xi_next = np.array(
            [
                [0.0, 0.0, +g_xi, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, +g_sigma],
                [0.0, 0.0, -g_xi, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -g_sigma],
            ]
        )
        h_xi_next = np.array(
            [
                self.max_dcm_height - self.ref_dcm[2],
                self.ref_dcm[2] - self.min_dcm_height,
            ]
        )

        G = np.vstack([G_cop, G_lambda, G_omega, G_xi_next])
        h = np.hstack([h_cop, h_lambda, h_omega, h_xi_next])

        P = np.diag([1e-6] * 7 + [1.0, 1.0, 1e-3])
        q = np.zeros(10)

        Delta_x = solve_qp(P, q, G, h, A, b, solver=self.solver)
        self.result = VHIPQPResult(Delta_x)  # saved for later
        # Delta_sigma=self.result.Delta_sigma
        # print(f"{Delta_sigma=}")

        Delta_omega_opt = Delta_x[3]
        Delta_r_opt = Delta_x[4:6]
        Delta_lambda_opt = Delta_x[6]
        self.omega = omega_d + Delta_omega_opt
        self.dcm = self.pendulum.com.p + self.pendulum.com.pd / self.omega
        return (Delta_r_opt, Delta_lambda_opt)
