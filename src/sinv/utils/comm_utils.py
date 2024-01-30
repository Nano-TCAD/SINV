"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2024-01

Copyright 2024 ETH Zurich. All rights reserved.
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


def aggregate_partitions(
    A_local: bsp,
    start_blockrows: list,
    partition_sizes: list,
    symmetry: str = None,
) -> bsp:
    """Aggregate the local partitions of a block tridiagonal matrix to the
    global matrix.

    Bridges blocks are taken from the partition that owns the corresponding
    blockrow.

    Parameters
    ----------
    A_local : bsp
        Local partitions of the global matrix.
    start_blockrows : list
        List of the indices of the first blockrow of each partition.
    partition_sizes : list
        List of the sizes of each partition.
    symmetry : str, optional
        Whether the global matrix is symmetric or not. The default is None.

    Returns
    -------
    A_global : bsp
        Global matrix in bparse format.
    """

    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()

    if comm_size != len(partition_sizes):
        raise ValueError("Number of partitions and number of processes do not match.")
    if comm_size == 1:
        raise ValueError("Too few processes (1).")

    comm_rank = comm.Get_rank()

    if comm_rank == 0:
        # Rank 0 aggregates partitions from other ranks.
        n_blocks = sum(partition_sizes)
        blocksize = A_local[0, 0].shape[0]

        diagonal_blocks = np.zeros(
            (n_blocks, blocksize, blocksize), dtype=A_local.dtype
        )
        upper_diagonal_blocks = np.zeros(
            (n_blocks - 1, blocksize, blocksize), dtype=A_local.dtype
        )
        lower_diagonal_blocks = np.zeros(
            (n_blocks - 1, blocksize, blocksize), dtype=A_local.dtype
        )

        # 1. Put rank 0 local partition in the global matrix.
        diagonal_blocks[0 : partition_sizes[0], :, :] = np.array(
            A_local.diagonal(offset=0)
        )[0 : partition_sizes[0], :, :]
        upper_diagonal_blocks[0 : partition_sizes[0], :, :] = np.array(
            A_local.diagonal(offset=1)
        )[0 : partition_sizes[0], :, :]
        lower_diagonal_blocks[0 : partition_sizes[0] - 1, :, :] = np.array(
            A_local.diagonal(offset=-1)
        )[0 : partition_sizes[0] - 1, :, :]

        # 2. Receive other ranks local partitions and put them in the global matrix.
        for i in range(1, comm_size):
            diagonal_blocks[
                start_blockrows[i] : start_blockrows[i] + partition_sizes[i], :, :
            ] = comm.recv(source=i, tag=0)

            upper_diagonal_blocks[
                start_blockrows[i] : start_blockrows[i] + partition_sizes[i], :, :
            ] = comm.recv(source=i, tag=1)

            lower_diagonal_blocks[
                start_blockrows[i] - 1 : start_blockrows[i] + partition_sizes[i] - 1,
                :,
                :,
            ] = comm.recv(source=i, tag=2)

        offsets = [0, 1, -1]
        A_global = bsp.BDIA(
            offsets,
            [diagonal_blocks, upper_diagonal_blocks, lower_diagonal_blocks],
            symmetry=symmetry,
        )

        return A_global

    else:
        # Other ranks send their local partition to rank 0.
        if comm_rank == comm_size - 1:
            diagonal_blocks = np.array(A_local.diagonal(offset=0))[1:, :, :]
            upper_diagonal_blocks = np.array(A_local.diagonal(offset=1))[1:, :, :]
            lower_diagonal_blocks = np.array(A_local.diagonal(offset=-1))[0:, :, :]
        else:
            diagonal_blocks = np.array(A_local.diagonal(offset=0))[1:-1, :, :]
            upper_diagonal_blocks = np.array(A_local.diagonal(offset=1))[1:, :, :]
            lower_diagonal_blocks = np.array(A_local.diagonal(offset=-1))[0:-1, :, :]

        comm.send(diagonal_blocks, dest=0, tag=0)
        comm.send(upper_diagonal_blocks, dest=0, tag=1)
        comm.send(lower_diagonal_blocks, dest=0, tag=2)

        return None


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

    # plt.matshow(bsparse_matrix_localpartition.toarray())
    # plt.title("Local partition n°" + str(comm.Get_rank()))
    # plt.show()

    global_matrix = aggregate_partitions(
        bsparse_matrix_localpartition,
        start_blockrows,
        partition_sizes,
    )

    if comm.Get_rank() == 0:
        plt.matshow(global_matrix.toarray())
        plt.title("Global matrix")
        plt.show()

        assert np.allclose(global_matrix.toarray(), bsparse_matrix.toarray())
