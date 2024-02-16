#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0
# Copyright 2024 Inria

from .process import Process


class Pusher(Process):
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

    def push(self, dv, gain=None):
        if gain is None:
            gain = self.gain
        for pendulum in self.pendulums:
            comd = pendulum.com.pd
            pendulum.com.set_vel(comd + dv)

    def step(self, dt):
        """Apply regular impulses to the inverted pendulum.

        Args:
            dt: Time step.
        """
        self.nb_ticks += 1
        one_sec = int(1.0 / dt)
        if self.started and self.nb_ticks % one_sec == 0:
            self.push()
