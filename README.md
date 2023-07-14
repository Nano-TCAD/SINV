# SelectedInversion
Implementations of several selected inversion algorithms.  

## $G^r$ implemented algorithms:
- [x] __RGF__ (Recursive Green's Function)
    - Serial, non-pivoting, tri-diagonal matrix selected inversion algorithm
    - @reference: https://doi.org/10.1063/1.1432117
- [x] __RGF 2-sided__ 
    - 2 Processes version of RGF
    - @reference: https://doi.org/10.1007/s10825-013-0458-7
- [x] __Tri-diagonal Gaussian elimination__
    - Serial, non-pivoting, tri-diagonal matrix selected inversion algorithm
    - @reference: https://doi.org/10.1016/j.jcp.2009.03.035
- [x] __BCR-S__ (Block Cyclic Reduction)
    - Serial, non-pivoting, block tri-diagonal matrix selected inversion algorithm. 
    - @reference: https://doi.org/10.1016/j.jcp.2009.03.035
    - @reference: https://doi.org/10.1017/CBO9780511812583
- [x] __BCR-P__
    - Multi-processes version of BCR-S
    - @reference: https://doi.org/10.1017/CBO9780511812583
- [x] __HPR-P__ (Hybrid Parallel Reccurence)
    - Multi-processes, Schur-complement based, block tri-diagonal matrix selected inversion algorithm.
    - @reference: https://doi.org/10.1016/j.jcp.2009.03.035
- [ ] __PW-S__ (Pairwise)
    - Serial
- [ ] __PW-P__ (Pairwise)
    - Multi-processes

## $G^<$ implemented algorithms:
None yet.