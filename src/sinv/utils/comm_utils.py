"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2024-01

Copyright 2024 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

from sinv.utils import rlcu

import bsparse as bsp

import numpy as np
from mpi4py import MPI


import matplotlib.pyplot as plt


def distribute_partitions_tridiagonals_matrix(
    A: bsp,
    start_blockrows: list,
    partition_sizes: list,
    symmetry: str = None,
) -> bsp:
    """Distribute the partitions of a block tridiagonal matrix to the processes.

    Parameters
    ----------
    A : bsp
        Global matrix in bparse format.
    start_blockrows : list
        List of the indices of the first blockrow of each partition.
    partition_sizes : list
        List of the sizes of each partition.
    is_symmetric : bool, optional
        Whether the global matrix is symmetric or not. The default is False.

    Returns
    -------
    A_local : bsp
        Local partitions of the global matrix.

    """

    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()

    if comm_size != len(partition_sizes):
        raise ValueError("Number of partitions and number of processes do not match.")

    comm_rank = comm.Get_rank()

    if comm_rank == 0:
        # Rank 0 sends partitions
        for i in range(1, comm_size):
            A_n = rlcu.slice_partition(
                A, start_blockrows[i], partition_sizes[i], include_bridges_blocks=True
            )

            diagonal_blocks = np.array(A_n.diagonal(offset=0))
            comm.send(diagonal_blocks, dest=i, tag=0)

            upper_diagonal_blocks = np.array(A_n.diagonal(offset=1))
            comm.send(upper_diagonal_blocks, dest=i, tag=1)

            lower_diagonal_blocks = np.array(A_n.diagonal(offset=-1))
            comm.send(lower_diagonal_blocks, dest=i, tag=2)

        # Rank 0 return it's own local partition
        return rlcu.slice_partition(
            A, start_blockrows[0], partition_sizes[0], include_bridges_blocks=True
        )

    else:
        # Others ranks receive partitions
        diagonal_blocks = comm.recv(source=0, tag=0)
        upper_diagonal_blocks = comm.recv(source=0, tag=1)
        lower_diagonal_blocks = comm.recv(source=0, tag=2)

        offsets = [0, 1, -1]
        A_local = bsp.BDIA(
            offsets,
            [diagonal_blocks, upper_diagonal_blocks, lower_diagonal_blocks],
            symmetry=symmetry,
        )

        return A_local


from sinv.utils import gmu
from sinv.utils import rlcu
from sinv.utils import partu


if __name__ == "__main__":
    n_blocks = 10
    blocksize = 2
    is_complex = False

    (
        diagonal_blocks,
        upper_diagonal_blocks,
        lower_diagonal_blocks,
    ) = gmu.create_block_tridiagonal_matrix(
        n_blocks, blocksize, is_complex, is_symmetric=False
    )

    bsparse_matrix = rlcu.block_tridiagonal_to_BDIA(
        diagonal_blocks,
        upper_diagonal_blocks,
        lower_diagonal_blocks,
        symmetry=None,
    )

    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()

    start_blockrows, partition_sizes, end_blockrows = partu.get_partitions_indices(
        comm_size,
        n_blocks,
    )

    bsparse_matrix_localpartition = distribute_partitions_tridiagonals_matrix(
        bsparse_matrix,
        start_blockrows,
        partition_sizes,
    )

    plt.matshow(bsparse_matrix_localpartition.toarray())
    plt.title("Local partition n°" + str(comm.Get_rank()))
    plt.show()
