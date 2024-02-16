#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Inria


from dataclasses import dataclass

from numpy.typing import NDArray


@dataclass
class Point:
    p: NDArray[float]
    pd: NDArray[float]
