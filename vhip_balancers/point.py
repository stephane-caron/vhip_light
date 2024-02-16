#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0
# Copyright 2024 Inria


from dataclasses import dataclass

from numpy.typing import NDArray


@dataclass
class Point:
    p: NDArray[float]
    pd: NDArray[float]

    def set_pos(self, p: NDArray[float]) -> None:
        self.p = p

    def set_vel(self, pd: NDArray[float]) -> None:
        self.pd = pd

    @property
    def x(self):
        return self.p[0]

    @property
    def y(self):
        return self.p[1]

    @property
    def z(self):
        return self.p[2]
