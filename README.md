# SINV : Selected Inversion Library
Implementations of several shared and distributed memory selected-inversion algorithms. These algorithms performs efficient inversion on block tridiagonal matrices.

## Solvers: $A \cdot X = B$:
### __RGF__ (Recursive Green's Function)
@reference: https://doi.org/10.1063/1.1432117  
- Serial, non-pivoting, block-tridiagonal matrix selected inversion algorithm.
    
### __RGF 2-sided__ 
@reference: https://doi.org/10.1007/s10825-013-0458-7  
- 2-processes parallel version of RGF.

### __BCR-Serial__ (Block Cyclic Reduction)
@reference: https://doi.org/10.1016/j.jcp.2009.03.035  
@reference: https://doi.org/10.1017/CBO9780511812583
- Serial, non-pivoting, block tri-diagonal matrix selected inversion algorithm. 

### __BCR-Parallel__ (Block Cyclic Reduction)
@reference: https://doi.org/10.1017/CBO9780511812583
- Multi-processes version of BCR-Serial.

### __PSR__ (Parallel Schur reduction)
@reference: https://doi.org/10.1016/j.jcp.2009.03.035
- Multi-processes, Schur-complement reduction based, block-tridiagonal matrix selected inversion algorithm.

### __PDIV/PairWise__ (P-Divisions)
@reference: https://doi.org/10.1063/1.2748621  
@reference: https://doi.org/10.1063/1.3624612  
@reference: https://doi.org/10.1007/978-3-319-78024-5_55  

#### pdiv_localmap:
- Specific implementation of PDIV/PairWise algorithm that leave the partitions on the processes and perform the update steps locally. Final results are scattered on the processes.
