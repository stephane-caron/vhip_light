#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0
# Copyright 2024 Inria

import numpy as np
import scipy.signal
from numpy.typing import NDArray

from .balancer import Balancer


class PolePlacementBalancer(Balancer):
    """Pole placement balancer.

    This is a "bonus" stabilizer, not reported in the paper, that was an
    intermediate step in our derivation of the VHIPQPBalancer.

    Args:
        pendulum: Inverted pendulum to stabilize.
        k_z: Feedback gain between DCM altitude and normalized leg stiffness
            input.

    Notes:
        This stabilizer also performs pole placement on a 4D DCM (using a
        velocity rather than position DCM though), but contrary to
        VHIPQPBalancer it doesn't force the closed-loop matrix to be
        diagonal. We started out exploring this stabilizer first.

        The first thing to observe by direct pole placement is that the gain
        matrix has essentially four non-zero gains in general. You can try out
        the :func:`set_poles` function to verify this.

        The closed-loop system with four gains has structure: in the horizontal
        plane it is equivalent to the VRPBalancer, and the normalized leg
        stiffness lambda depends on both the vertical DCM and the natural
        frequency omega. We observed that this system performs identically to
        the previous one in the horizontal plane, and always worse than the
        previous one vertically.

        However, raising the k_z (vertical DCM to lambda) gain to large values,
        we noticed that the vertical tracking of this stabilizer converged to
        that of the VRPBalancer. In the limit where k_z goes to infinity, the
        system slides on the constraint given by Equation (21) in the paper.
        This is how
    we came to the derivation of the VHIPQPBalancer.
    """

    def __init__(self, pendulum, k_z, kp:float):
        super(PolePlacementBalancer, self).__init__(pendulum)

        ref_dcm = self.ref_comd + self.ref_omega * self.ref_com
        # ref_cop = np.zeros(3)  # assumption of this stabilizer
        assert np.linalg.norm(self.contact.R - np.eye(3)) < 1e-5
        A = np.array(
            [
                [self.ref_omega, 0.0, 0.0, ref_dcm[0]],
                [0.0, self.ref_omega, 0.0, ref_dcm[1]],
                [0.0, 0.0, self.ref_omega, ref_dcm[2]],
                [0.0, 0.0, 0.0, 2.0 * self.ref_omega],
            ]
        )
        B = -np.array(
            [
                [self.ref_lambda, 0.0, self.ref_cop[0]],
                [0.0, self.ref_lambda, self.ref_cop[1]],
                [0.0, 0.0, self.ref_cop[2]],
                [0.0, 0.0, 1.0],
            ]
        )
        self.A = A
        self.B = B
        self.K = None  # call set_gains or set_poles
        self.kp = kp
        self.ref_dcm = ref_dcm
        #
        self.set_critical_gains(k_z)

    def set_poles(self, poles: NDArray[float]):
        """Place poles using SciPy's implementation of Kautsky et al.

        Args:
            poles: Desired poles of the closed-loop system.
        """
        bunch = scipy.signal.place_poles(self.A, self.B, poles)
        self.K = -bunch.gain_matrix  # place_poles assumes A - B * K

    def set_gains(self, gains):
        """Set gains from 4D DCM error to 3D input ``[zmp_x, zmp_y, lambda]``.

        Parameters
        ----------
        gains : (4,) array
            List of gains ``[k_x, k_y, k_z, k_omega]``.
        """
        k_x, k_y, k_z, k_omega = gains
        self.K = np.array(
            [
                [k_x, 0.0, 0.0, 0.0],
                [0.0, k_y, 0.0, 0.0],
                [0.0, 0.0, k_z, k_omega],
            ]
        )

    def set_critical_gains(self, k_z):
        """Set critical gain `k_omega` for a desired vertical DCM gain `k_z`.

        Args:
            k_z: Desired vertical DCM to normalized leg stiffness gain.
        """
        assert k_z > 1e-10, "Feedback gain needs to be positive"
        omega = self.ref_omega
        k_xy = self.kp / omega
        gamma = omega * self.kp
        k_omega = omega + (k_z * self.ref_dcm[2] + gamma**2) / gamma
        self.set_gains([k_xy, k_xy, k_z, k_omega])

    def compute_compensation(self, dt: float):
        """Compute CoP and normalized leg stiffness compensation."""
        omega = self.omega
        com = self.pendulum.com.p
        comd = self.pendulum.com.pd
        dcm = comd + omega * com
        Delta_omega = omega - self.ref_omega
        Delta_x = np.array(
            [
                dcm[0] - self.ref_dcm[0],
                dcm[1] - self.ref_dcm[1],
                dcm[2] - self.ref_dcm[2],
                Delta_omega,
            ]
        )
        Delta_u = np.dot(self.K, Delta_x)
        Delta_lambda = Delta_u[2]
        Delta_r = Delta_u[:2]  # contact is horizontal for now
        omegad = 2 * self.ref_omega * Delta_omega - Delta_lambda
        self.omega += omegad * dt
        self.dcm = com + comd / omega
        return (Delta_r, Delta_lambda)
