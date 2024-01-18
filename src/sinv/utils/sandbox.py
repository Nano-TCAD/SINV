import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

from read_load_convert_utils import read_block_tridiagonal_matrix, read_local_block_tridiagonal_partition, block_tridiagonal_to_bsparse
from testing_utils import create_block_tridiagonal_matrix, save_block_tridigonal_matrix
from partition_utils import get_local_partition_indices


# if __name__ == "__main__":
    
#     n_blocks = 10
#     blocksize = 4
#     is_complex = False
    
#     diagonal_blocks, upper_diagonal_blocks, lower_diagonal_blocks = create_block_tridiagonal_matrix(n_blocks, blocksize, is_complex)
        
#     save_block_tridigonal_matrix(diagonal_blocks, upper_diagonal_blocks, lower_diagonal_blocks, "test.npz")
    
#     loaded_diagonal_blocks, loaded_upper_diagonal_blocks, loaded_lower_diagonal_blocks = read_block_tridiagonal_matrix("test.npz")
    
#     np.allclose(diagonal_blocks, loaded_diagonal_blocks)
#     np.allclose(upper_diagonal_blocks, loaded_upper_diagonal_blocks)
#     np.allclose(lower_diagonal_blocks, loaded_lower_diagonal_blocks)
   

   
# if __name__ == "__main__":
    
#     n_blocks = 10
#     blocksize = 1
#     is_complex = False
    
#     diagonal_blocks, upper_diagonal_blocks, lower_diagonal_blocks = create_block_tridiagonal_matrix(n_blocks, blocksize, is_complex)
        
#     save_block_tridigonal_matrix(diagonal_blocks, upper_diagonal_blocks, lower_diagonal_blocks, "test.npz")
     
#     partition_number = 0
#     n_partitions = 4
#     partitions_blocksizes = [3, 2, 2, 3]
#     bridges_blocks = True
    
#     start_blockrow, end_blockrow, partition_size = get_local_partition_indices(partition_number, n_partitions, partitions_blocksizes)
    
#     loaded_diagonal_blocks, loaded_upper_diagonal_blocks, loaded_lower_diagonal_blocks = read_local_block_tridiagonal_partition("test.npz", start_blockrow, partition_size, blocksize, include_bridges_blocks=bridges_blocks)
    
#     fig, axs = plt.subplots(3, 2)
#     axs[0, 0].matshow(diagonal_blocks)
#     axs[0, 1].matshow(loaded_diagonal_blocks)
#     axs[1, 0].matshow(upper_diagonal_blocks)
#     axs[1, 1].matshow(loaded_upper_diagonal_blocks)
#     axs[2, 0].matshow(lower_diagonal_blocks)
#     axs[2, 1].matshow(loaded_lower_diagonal_blocks)
#     plt.show()





if __name__ == "__main__":
    n_blocks = 10
    blocksize = 4
    is_complex = False
    
    diagonal_blocks, upper_diagonal_blocks, lower_diagonal_blocks = create_block_tridiagonal_matrix(n_blocks, blocksize, is_complex)
    
    bdia_matrix = block_tridiagonal_to_bsparse(diagonal_blocks, upper_diagonal_blocks, lower_diagonal_blocks, blocksize)
    
    plt.matshow(bdia_matrix.toarray())
    plt.show()
    
    