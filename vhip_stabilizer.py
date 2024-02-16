#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2016-2019 CNRS-UM LIRMM
# Copyright 2024 Inria

"""This example compares two stabilizers for the inverted pendulum model. The
first one (baseline) is based on proportional feedback of the 3D DCM
[Englsberger15]_. The second one (proposed) performs proportional feedback of a
4D DCM of the same model [Caron20]_.
"""

import sys
from warnings import warn

import IPython
import matplotlib
import matplotlib.pylab as plt
import numpy as np
import scipy.signal
from numpy.typing import NDArray

from vhip_light import Point

try:
    import cvxpy
except ImportError:
    raise ImportError("This example requires CVXPY, install it e.g. via pip")

from numpy import array, dot, eye, hstack, sqrt, vstack, zeros
from qpsolvers import solve_qp

GRAVITY = [0.0, 0.0, -9.81]  # [m] / [s]^(-2)
MASS = 38.0  # [kg]

MAX_DCM_HEIGHT = 1.0  # [m]
MIN_DCM_HEIGHT = 0.5  # [m]

MAX_FORCE = 1000.0  # [N]
MIN_FORCE = 1.0  # [N]

REF_OFFSET = array([0.0, 0.0, 0.0])  # [m]
K_P = 3.0  # proportional DCM feedback gain

assert K_P > 1.0, "DCM feedback gain needs to be greater than one"


class Contact:
    pass


class InvertedPendulum:

    def __init__(
        self,
        pos,
        vel,
        contact,
        lambda_min=1e-5,
        lambda_max=None,
        clamp=True,
        color="b",
        size=0.02,
    ):
        super(InvertedPendulum, self).__init__()
        com = Point(pos, vel)
        self.clamp = clamp
        self.color = color
        self.com = com
        self.contact = contact
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
            cop_local = dot(self.contact.R.T, cop - self.contact.p)
            if cop_local[0] >= self.contact.shape[0]:
                cop_local[0] = self.contact.shape[0] - 1e-5
            elif cop_local[0] <= -self.contact.shape[0]:
                cop_local[0] = -self.contact.shape[0] + 1e-5
            if cop_local[1] >= self.contact.shape[1]:
                cop_local[1] = self.contact.shape[1] - 1e-5
            elif cop_local[1] <= -self.contact.shape[1]:
                cop_local[1] = -self.contact.shape[1] + 1e-5
            cop = self.contact.p + dot(self.contact.R, cop_local)
        elif __debug__:
            cop_check = dot(self.contact.R.T, cop - self.contact.p)
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
        omega = sqrt(self.lambda_)
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


class Stabilizer:
    """Base class for stabilizer processes.

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

    def __init__(self, pendulum):
        super(Stabilizer, self).__init__()
        ref_com = pendulum.com.p + REF_OFFSET
        n = pendulum.contact.normal
        lambda_ = -dot(n, GRAVITY) / dot(n, ref_com - pendulum.contact.p)
        omega = sqrt(lambda_)
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

    def on_tick(self, sim):
        """Set inverted pendulum CoP and stiffness inputs.

        Parameters
        ----------
        sim : pymanoid.Simulation
            Simulation instance.
        """
        Delta_r, Delta_lambda = self.compute_compensation()
        cop = self.ref_cop + dot(self.contact.R[:3, :2], Delta_r)
        lambda_ = self.ref_lambda + Delta_lambda
        self.pendulum.set_cop(cop)
        self.pendulum.set_lambda(lambda_)


class VRPStabilizer(Stabilizer):
    """Inverted pendulum stabilizer based on proportional feedback of the
    3D divergent component of motion (DCM) applied to the virtual repellent
    point (VRP).

    Parameters
    ----------
    pendulum : pymanoid.models.InvertedPendulum
        Inverted pendulum to stabilize.

    Attributes:
    ----------
    ref_dcm : (3,) array
        Desired (3D) divergent component of motion.
    ref_vrp : (3,) array
        Desired virtual repellent point (VRP).

    Notes:
    -----
    See "Three-Dimensional Bipedal Walking Control Based on Divergent Component
    of Motion" (Englsberger et al., IEEE Transactions on Robotics) for details.
    """

    def __init__(self, pendulum):
        super(VRPStabilizer, self).__init__(pendulum)
        self.ref_dcm = self.ref_com
        self.ref_vrp = self.ref_com

    def compute_compensation(self):
        """Compute CoP and normalized leg stiffness compensation."""
        omega = self.omega
        com = self.pendulum.com.p
        comd = self.pendulum.com.pd
        dcm = com + comd / omega
        Delta_dcm = dcm - self.ref_dcm
        vrp = self.ref_vrp + K_P * Delta_dcm
        n = self.pendulum.contact.n
        gravito_inertial_force = omega**2 * (com - vrp) - GRAVITY
        displacement = com - self.pendulum.contact.p
        lambda_ = dot(n, gravito_inertial_force) / dot(n, displacement)
        cop = com - gravito_inertial_force / lambda_
        Delta_r = dot(self.contact.R.T, cop - self.ref_cop)[:2]
        Delta_lambda = lambda_ - self.ref_lambda
        self.dcm = dcm
        return (Delta_r, Delta_lambda)


class VHIPStabilizer(Stabilizer):
    """Proportional feedback of the 4D DCM of the VHIP.

    Parameters
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

    def __init__(self, pendulum):
        super(VHIPStabilizer, self).__init__(pendulum)
        r_d_contact = dot(self.contact.R.T, self.ref_cop - self.contact.p)[:2]
        self.r_contact_max = array(self.contact.shape)
        self.ref_cop_contact = r_d_contact
        self.ref_dcm = self.ref_com
        self.ref_vrp = self.ref_com

    def compute_compensation(self):
        """Compute CoP and normalized leg stiffness compensation."""
        # Measurements
        Delta_com = self.pendulum.com.p - self.ref_com
        Delta_comd = self.pendulum.com.pd - self.ref_comd
        height = dot(self.contact.normal, self.pendulum.com.p - self.contact.p)
        lambda_d = self.ref_lambda
        measured_comd = self.pendulum.com.pd
        nu_d = self.ref_vrp
        omega_d = self.ref_omega
        r_d_contact = self.ref_cop_contact
        xi_d = self.ref_dcm

        # Force limits
        lambda_max = MAX_FORCE / (MASS * height)
        lambda_min = MIN_FORCE / (MASS * height)
        omega_max = sqrt(lambda_max)
        omega_min = sqrt(lambda_min)

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
        Delta_r_world = contact.R[:3, :2] * Delta_r
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
                Delta_xid == lambda_d / omega_d * ((1 - K_P) * Delta_xi + u),
                Delta_omegad == omega_d * (1 - K_P) * Delta_omega,
                Delta_nu
                == Delta_r_world + GRAVITY * Delta_lambda / lambda_d**2,
                cvxpy.abs(r_contact) <= self.r_contact_max,
                lambda_ <= lambda_max,
                lambda_ >= lambda_min,
                xi_z <= MAX_DCM_HEIGHT,
                xi_z >= MIN_DCM_HEIGHT,
                omega <= omega_max,
                omega >= omega_min,
            ],
        )
        prob.solve()

        # Read outputs from solution
        Delta_lambda_opt = Delta_lambda.value
        Delta_r_opt = array(Delta_r.value).reshape((2,))
        self.omega = omega_d + Delta_omega.value
        self.dcm = self.pendulum.com.p + self.pendulum.com.pd / self.omega
        return (Delta_r_opt, Delta_lambda_opt)


class VHIPQPStabilizer(VHIPStabilizer):
    """Proportional feedback of the 4D DCM of the VHIP.

    Args:
        pendulum: Inverted pendulum to stabilize.

    Notes:
        This implementation transcripts QP matrices from
        :class:`VHIPStabilizer`. We checked that the two produce the same
        outputs before switching to C++ in
        <https://github.com/stephane-caron/vhip_walking_controller/>. (This
        step would not have been necessary if we had a modeling language for
        convex optimization directly in C++.)
    """

    def compute_compensation(self):
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
        height = dot(self.contact.normal, self.pendulum.com.p - self.contact.p)
        lambda_max = MAX_FORCE / (MASS * height)
        lambda_min = MIN_FORCE / (MASS * height)
        omega_max = sqrt(lambda_max)
        omega_min = sqrt(lambda_min)

        A = vstack(
            [
                hstack(
                    [
                        -K_P * eye(3),
                        (xi_d - nu_d).reshape((3, 1)) / omega_d,
                        self.contact.R[:3, :2],
                        (r_d - xi_d).reshape((3, 1)) / lambda_d,
                        eye(3),
                    ]
                ),
                hstack(
                    [
                        eye(3),
                        measured_comd.reshape((3, 1)) / omega_d**2,
                        zeros((3, 2)),
                        zeros((3, 1)),
                        zeros((3, 3)),
                    ]
                ),
                hstack(
                    [
                        zeros((1, 3)),
                        omega_d * (1 + K_P) * eye(1),
                        zeros((1, 2)),
                        -1 * eye(1),
                        zeros((1, 3)),
                    ]
                ),
            ]
        )
        b = hstack([zeros(3), Delta_com + Delta_comd / omega_d, zeros(1)])

        G_cop = array(
            [
                [0.0, 0.0, 0.0, 0.0, +1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, +1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        h_cop = array(
            [
                self.contact.shape[0] - r_d_contact[0],
                self.contact.shape[0] + r_d_contact[0],
                self.contact.shape[1] - r_d_contact[1],
                self.contact.shape[1] + r_d_contact[1],
            ]
        )

        G_lambda = array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, +1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
            ]
        )
        h_lambda = array([lambda_max - lambda_d, lambda_d - lambda_min])

        G_omega = array(
            [
                [0.0, 0.0, 0.0, +1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        h_omega = array([omega_max - omega_d, omega_d - omega_min])

        g_sigma = 1.5 * lambda_d * dt / omega_d
        g_xi = 1 + g_sigma * (1 - K_P)
        G_xi_next = array(
            [
                [0.0, 0.0, +g_xi, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, +g_sigma],
                [0.0, 0.0, -g_xi, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -g_sigma],
            ]
        )
        h_xi_next = array(
            [
                MAX_DCM_HEIGHT - self.ref_dcm[2],
                self.ref_dcm[2] - MIN_DCM_HEIGHT,
            ]
        )

        G = vstack([G_cop, G_lambda, G_omega, G_xi_next])
        h = hstack([h_cop, h_lambda, h_omega, h_xi_next])

        P = np.diag([1e-6] * 7 + [1.0, 1.0, 1e-3])
        q = np.zeros(10)

        Delta_x = solve_qp(P, q, G, h, A, b, solver="quadprog")
        Delta_omega_opt = Delta_x[3]
        Delta_r_opt = Delta_x[4:6]
        Delta_lambda_opt = Delta_x[6]
        self.omega = omega_d + Delta_omega_opt
        self.dcm = self.pendulum.com.p + self.pendulum.com.pd / self.omega
        return (Delta_r_opt, Delta_lambda_opt)


class BonusPolePlacementStabilizer(Stabilizer):
    """This is a "bonus" stabilizer, not reported in the paper, that was an
    intermediate step in our derivation of the VHIPQPStabilizer.

    Args:
        pendulum: Inverted pendulum to stabilize.
        k_z: Feedback gain between DCM altitude and normalized leg stiffness
            input.

    Notes:
        This stabilizer also performs pole placement on a 4D DCM (using a
        velocity rather than position DCM though), but contrary to
        VHIPQPStabilizer it doesn't force the closed-loop matrix to be
        diagonal. We started out exploring this stabilizer first.

        The first thing to observe by direct pole placement is that the gain
        matrix has essentially four non-zero gains in general. You can try out
        the :func:`set_poles` function to verify this.

        The closed-loop system with four gains has structure: in the horizontal
        plane it is equivalent to the VRPStabilizer, and the normalized leg
        stiffness lambda depends on both the vertical DCM and the natural
        frequency omega. We observed that this system performs identically to
        the previous one in the horizontal plane, and always worse than the
        previous one vertically.

        However, raising the k_z (vertical DCM to lambda) gain to large values,
        we noticed that the vertical tracking of this stabilizer converged to
        that of the VRPStabilizer. In the limit where k_z goes to infinity, the
        system slides on the constraint given by Equation (21) in the paper.
        This is how
    we came to the derivation of the VHIPQPStabilizer.
    """

    def __init__(self, pendulum, k_z):
        super(BonusPolePlacementStabilizer, self).__init__(pendulum)
        ref_dcm = self.ref_comd + self.ref_omega * self.ref_com
        # ref_cop = np.zeros(3)  # assumption of this stabilizer
        assert np.linalg.norm(self.contact.R - np.eye(3)) < 1e-5
        A = array(
            [
                [self.ref_omega, 0.0, 0.0, ref_dcm[0]],
                [0.0, self.ref_omega, 0.0, ref_dcm[1]],
                [0.0, 0.0, self.ref_omega, ref_dcm[2]],
                [0.0, 0.0, 0.0, 2.0 * self.ref_omega],
            ]
        )
        B = -array(
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
        self.K = array(
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
        k_xy = K_P / omega
        gamma = omega * K_P
        k_omega = omega + (k_z * self.ref_dcm[2] + gamma**2) / gamma
        self.set_gains([k_xy, k_xy, k_z, k_omega])

    def compute_compensation(self):
        """Compute CoP and normalized leg stiffness compensation."""
        omega = self.omega
        com = self.pendulum.com.p
        comd = self.pendulum.com.pd
        dcm = comd + omega * com
        Delta_omega = omega - self.ref_omega
        Delta_x = array(
            [
                dcm[0] - self.ref_dcm[0],
                dcm[1] - self.ref_dcm[1],
                dcm[2] - self.ref_dcm[2],
                Delta_omega,
            ]
        )
        Delta_u = dot(self.K, Delta_x)
        Delta_lambda = Delta_u[2]
        Delta_r = Delta_u[:2]  # contact is horizontal for now
        omegad = 2 * self.ref_omega * Delta_omega - Delta_lambda
        self.omega += omegad * dt
        self.dcm = com + comd / omega
        return (Delta_r, Delta_lambda)


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
        self.mask = array([1.0, 1.0, 1.0])
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

    def push(self, dv=None, gain=None, mask=None):
        if gain is None:
            gain = self.gain
        if dv is None:
            dv = 2.0 * np.random.random(3) - 1.0
            if self.mask is not None:
                dv *= self.mask
            dv *= gain / np.linalg.norm(dv)
            print("Pusher: dv = {}".format(repr(dv)))
        for pendulum in self.pendulums:
            comd = pendulum.com.pd
            pendulum.com.set_vel(comd + dv)


class Plotter:

    def __init__(self, stabilizers):
        super(Plotter, self).__init__()
        self.plots = {
            "omega": [[] for stab in stabilizers],
            "xi_x": [[] for stab in stabilizers],
            "xi_y": [[] for stab in stabilizers],
            "xi_z": [[] for stab in stabilizers],
        }
        self.stabilizers = stabilizers

    def on_tick(self, sim):
        for i, stab in enumerate(self.stabilizers):
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
        ref_omega = vrp_stabilizer.ref_omega
        ref_lambda = vrp_stabilizer.ref_lambda
        ref_dcm_p = vrp_stabilizer.ref_dcm
        refs = {
            "omega": [ref_omega**2, ref_lambda],
            "xi_x": [ref_dcm_p[0]],
            "xi_y": [ref_dcm_p[1]],
            "xi_z": [ref_dcm_p[2]],
        }
        for figid, figname in enumerate(self.plots):
            plt.subplot(411 + figid)
            for i, stab in enumerate(self.stabilizers):
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
    stabilizers = []

    pendulums.append(
        InvertedPendulum(init_pos, init_vel, contact, color="b", size=0.019)
    )
    vhip_stabilizer = VHIPQPStabilizer(pendulums[-1])
    stabilizers.append(vhip_stabilizer)

    pendulums.append(
        InvertedPendulum(init_pos, init_vel, contact, color="g", size=0.02)
    )
    vrp_stabilizer = VRPStabilizer(pendulums[-1])
    stabilizers.append(vrp_stabilizer)

    if "--bonus" in sys.argv:
        pendulums.append(
            InvertedPendulum(
                init_pos, init_vel, contact, color="r", size=0.015
            )
        )
        bonus_stabilizer = BonusPolePlacementStabilizer(pendulums[-1], k_z=100)
        stabilizers.append(bonus_stabilizer)

    pusher = Pusher(pendulums)
    plotter = Plotter(stabilizers)

    sim = []
    for stabilizer, pendulum in zip(stabilizers, pendulums):
        sim.append(stabilizer)  # before pendulum
        sim.append(pendulum)
    sim.append(plotter)  # before pusher
    sim.append(pusher)

    def reset():
        for stab in stabilizers:
            stab.reset_pendulum()
        sim.step()

    sim.step(42)  # go to reference
    impulse = array([0.0, -0.09, 0.0])
    # push_three_times()  # scenario for Fig. 1 of the paper
    # record_video()  # video for v1 of the paper
    reset()

    print(
        """

Variable-Height Inverted Pendulum Stabilization
===============================================

Ready to go! You can access all state variables via this IPython shell.
Here is the list of global objects. Use <TAB> to see what's inside.

    pendulums -- LIP and VHIP inverted pendulum states
    stabilizers -- their respective balance feedback controllers
    pusher -- applies external impulse to both pendulums at regular intervals
    plotter -- logs plot data

Call ``plotter.plot()`` to draw a LIP/VHIP comparison plot (Fig. 2 of the
manuscript).

You can pause/resume processes or the whole simulation by:

    sim.start() -- start/resume simulation
    sim.step(100) -- run simulation in current thread for 100 steps
    sim.stop() -- stop/pause simulation
    pusher.push([0., 0.12, 0.]) -- apply same impulse to both pendulums
    reset() -- reset both inverted pendulums to the origin

If a pendulum diverges, both pendulums will eventually disappear from the GUI.
          Call ``reset()`` in this case.

Enjoy :)

"""
    )

    if IPython.get_ipython() is None:  # give the user a prompt
        IPython.embed()
    IPython.get_ipython().magic("pylab")  # for plots
