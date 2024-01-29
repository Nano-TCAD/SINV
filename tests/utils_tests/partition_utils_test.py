"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2024-01

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

from sinv.utils import partu

import pytest


@pytest.mark.parametrize(
    "n_partitions, total_size, partitions_distribution, solution_start_blockrows, solution_partition_sizes, solution_end_blockrows",
    [
        (3, 10, None, [0, 4, 7], [4, 3, 3], [4, 7, 10]),
        (3, 10, [40, 30, 30], [0, 4, 7], [4, 3, 3], [4, 7, 10]),
        (3, 10, [10, 10, 80], [0, 1, 2], [1, 1, 8], [1, 2, 10]),
    ],
)
def test_get_partitions_indices(
    n_partitions: int,
    total_size: int,
    partitions_distribution: list,
    solution_start_blockrows: list,
    solution_partition_sizes: list,
    solution_end_blockrows: list,
):
    start_blockrows, partition_sizes, end_blockrows = partu.get_partitions_indices(
        n_partitions, total_size, partitions_distribution
    )

    assert start_blockrows == solution_start_blockrows
    assert partition_sizes == solution_partition_sizes
    assert end_blockrows == solution_end_blockrows


@pytest.mark.parametrize(
    "n_partitions, total_size, partitions_distribution, solution_start_blockrows, solution_partition_sizes, solution_end_blockrows",
    [
        (3, 10, None, [0, 4, 7], [4, 3, 3], [4, 7, 10]),
        (3, 10, [40, 30, 30], [0, 4, 7], [4, 3, 3], [4, 7, 10]),
        (3, 10, [10, 10, 80], [0, 1, 2], [1, 1, 8], [1, 2, 10]),
    ],
)
@pytest.mark.parametrize(
    "current_partition",
    [
        pytest.param(0, id="p0"),
        pytest.param(1, id="p1"),
        pytest.param(2, id="p2"),
    ],
)
def test_get_local_partition_indices(
    n_partitions: int,
    total_size: int,
    partitions_distribution: list,
    solution_start_blockrows: list,
    solution_partition_sizes: list,
    solution_end_blockrows: list,
    current_partition: int,
):
    start_blockrow, partition_size, end_blockrow = partu.get_local_partition_indices(
        current_partition, n_partitions, total_size, partitions_distribution
    )

    assert start_blockrow == solution_start_blockrows[current_partition]
    assert partition_size == solution_partition_sizes[current_partition]
    assert end_blockrow == solution_end_blockrows[current_partition]
