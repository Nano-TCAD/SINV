# SelectedInversion
Implementations of several selected-blocks-inversion algorithms. These algorithms performs efficient inversion on block tridiagonal matrices.

## Solvers: $G^r = A^{-1}$:
### __RGF__ (Recursive Green's Function)
@reference: https://doi.org/10.1063/1.1432117  
- Serial, non-pivoting, block-tridiagonal matrix selected inversion algorithm.
    
### __RGF 2-sided__ 
@reference: https://doi.org/10.1007/s10825-013-0458-7  
- 2-processes parallel version of RGF.

### __Block-Tridiagonal LU solve__
@reference: https://doi.org/10.1016/j.jcp.2009.03.035
- Serial, non-pivoting, block-tridiagonal matrix selected inversion algorithm.

### __BCR-Serial__ (Block Cyclic Reduction)
@reference: https://doi.org/10.1016/j.jcp.2009.03.035  
@reference: https://doi.org/10.1017/CBO9780511812583
- Serial, non-pivoting, block tri-diagonal matrix selected inversion algorithm. 

### __BCR-Parallel__ (Block Cyclic Reduction)
@reference: https://doi.org/10.1017/CBO9780511812583
- Multi-processes version of BCR-Serial.

### __PSR__ (Parallel Schur reduction) (ex. hybrid)
@reference: https://doi.org/10.1016/j.jcp.2009.03.035
- Multi-processes, Schur-complement reduction based, block-tridiagonal matrix selected inversion algorithm.

### __PDIV/PairWise__ (P-Divisions)
@reference: https://doi.org/10.1063/1.2748621  
@reference: https://doi.org/10.1063/1.3624612  
@reference: https://doi.org/10.1007/978-3-319-78024-5_55  

#### pdiv_mincom:
- Specific implementation of PDIV/PairWise algorithm that minimise the communication between processes. It perform parallel partition inverse and fully aggregate the results on the root process before performing the update steps.

#### pdiv_aggregate:
- Specific implementation of PDIV/PairWise algorithm that aggregate and update the partitions in a divide-and-conquer fashion. Final results is fully aggregated on the root process.

#### pdiv_localmap:
- Specific implementation of PDIV/PairWise algorithm that leave the partitions on the processes and perform the update steps locally. Final results are distributed on the processes.

### Algorithms capabilities:

<table class="tg">
<thead>
  <tr>
    <th class="tg-7btt" colspan="2">Algorithms</th>
    <th class="tg-7btt">Reel-valued matrix</th>
    <th class="tg-7btt">Complex-valued matrix</th>
    <th class="tg-7btt">Symmetric matrix</th>
    <th class="tg-7btt">Non-symmetric matrix</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky" rowspan="3">RGF</td>
    <td class="tg-f8tv">serial left-to-right</td>
    <td class="tg-c3ow">✓</td>
    <td class="tg-c3ow">✓</td>
    <td class="tg-c3ow">✓</td>
    <td class="tg-c3ow">⤫</td>
  </tr>
  <tr>
    <td class="tg-f8tv">serial right-to-left</td>
    <td class="tg-zw5y">✓</td>
    <td class="tg-c3ow">✓</td>
    <td class="tg-c3ow">✓</td>
    <td class="tg-c3ow">⤫</td>
  </tr>
  <tr>
    <td class="tg-f8tv">2-sided</td>
    <td class="tg-zw5y">✓</td>
    <td class="tg-c3ow">✓</td>
    <td class="tg-c3ow">✓</td>
    <td class="tg-c3ow">⤫</td>
  </tr>
  <tr>
    <td class="tg-0pky" colspan="2">Block tridiagonal LU solve</td>
    <td class="tg-c3ow">✓</td>
    <td class="tg-c3ow">✓</td>
    <td class="tg-c3ow">✓</td>
    <td class="tg-c3ow">✓</td>
  </tr>
  <tr>
    <td class="tg-0pky" colspan="2">PSR (Parallel Schur reduction) (ex. hybrid)</td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="2">BCR (Block cyclic reduction)</td>
    <td class="tg-f8tv">serial</td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
  </tr>
  <tr>
    <td class="tg-f8tv">parallel</td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="3">PDIV / PairWise</td>
    <td class="tg-f8tv">mincom</td>
    <td class="tg-c3ow">✓</td>
    <td class="tg-c3ow">✓</td>
    <td class="tg-c3ow">✓</td>
    <td class="tg-c3ow">✓</td>
  </tr>
  <tr>
    <td class="tg-f8tv">aggregate</td>
    <td class="tg-c3ow">✓</td>
    <td class="tg-c3ow">✓</td>
    <td class="tg-c3ow">✓</td>
    <td class="tg-c3ow">✓</td>
  </tr>
  <tr>
    <td class="tg-f8tv">localmap</td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
  </tr>
</tbody>
</table>





## Solvers: $G^< = A^{-1} \Sigma^{<} G^{a}$:
Work in progress..