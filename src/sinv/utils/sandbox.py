import numpy as np
import matplotlib.pyplot as plt

from read_load_convert_utils import read_block_tridiagonal_matrix
from testing_utils import create_block_tridiagonal_matrix, save_block_tridigonal_matrix
from partition_utils import get_local_partition_indices

if __name__ == "__main__":
    
    n_blocks = 10
    blocksize = 4
    is_complex = False
    
    diagonal_blocks, upper_diagonal_blocks, lower_diagonal_blocks = create_block_tridiagonal_matrix(n_blocks, blocksize, is_complex)
        
    save_block_tridigonal_matrix(diagonal_blocks, upper_diagonal_blocks, lower_diagonal_blocks, "test.npz")
    
    # loaded_diagonal_blocks, loaded_upper_diagonal_blocks, loaded_lower_diagonal_blocks = read_block_tridiagonal_matrix("test.npz", n_blocks, blocksize, is_complex)
    
    # np.allclose(diagonal_blocks, loaded_diagonal_blocks)
    # np.allclose(upper_diagonal_blocks, loaded_upper_diagonal_blocks)
    # np.allclose(lower_diagonal_blocks, loaded_lower_diagonal_blocks)
    
    partition_number = 0
    n_partitions = 4
    partitions_blocksizes = [3, 2, 2, 3]
    
    start_blockrow, end_blockrow, partition_size = get_local_partition_indices(partition_number, n_partitions, partitions_blocksizes)
    
    print(start_blockrow, end_blockrow, partition_size)