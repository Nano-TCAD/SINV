"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2024-01

Copyright 2023-2024 ETH Zurich. All rights reserved.
"""

from sinv.algorithms.rgf2s import rgf2sided_base as rgf2s_base

import bsparse as bsp

import numpy as np
from mpi4py import MPI


def rgf2sided_commpartitions(
    A_global: bsp,
    symmetry: str = None,
) -> bsp:
    pass
