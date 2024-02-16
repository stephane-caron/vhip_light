#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0
# Copyright 2024 Inria

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class Contact:
    shape: Tuple[float, float]
    pos: NDArray[float]

    @property
    def p(self) -> NDArray[float]:
        return self.pos

    @property
    def z(self) -> float:
        return self.pos[2]

    @property
    def normal(self) -> NDArray[float]:
        """Normal vector in the inertial frame."""
        return np.array([0.0, 0.0, 1.0])

    @property
    def n(self) -> NDArray[float]:
        """Normal vector in the inertial frame."""
        return self.normal

    @property
    def R(self) -> NDArray[float]:
        """Rotation from the contact frame to the inertial frame."""
        return np.eye(3)
