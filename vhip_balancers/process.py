#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0
# Copyright 2024 Inria

import abc


class Process(abc.ABC):
    @abc.abstractmethod
    def step(self, dt: float) -> None:
        """Step process.

        Args:
            dt: Time step.
        """
