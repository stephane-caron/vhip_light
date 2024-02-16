#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0
# Copyright 2024 Inria

"""Variable-height inverted pendulum balancing in Python."""

from .contact import Contact
from .inverted_pendulum import InvertedPendulum
from .plotter import Plotter
from .point import Point
from .pole_placement_balancer import PolePlacementBalancer
from .process import Process
from .pusher import Pusher
from .vhip_balancer import VHIPBalancer
from .vhip_qp_balancer import VHIPQPBalancer
from .vrp_balancer import VRPBalancer

__version__ = "1.0.0"

__all__ = [
    "Contact",
    "InvertedPendulum",
    "Plotter",
    "Point",
    "PolePlacementBalancer",
    "Process",
    "Pusher",
    "VHIPBalancer",
    "VHIPQPBalancer",
    "VRPBalancer",
]
