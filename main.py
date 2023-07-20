"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-05

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import utils.generateMatrices    as genMat
import utils.convertMatrices     as convMat
import utils.transformMatrices   as transMat
import utils.vizualisation       as vizu

import algorithms.fullInversion as inv
import algorithms.rgf           as rgf
import algorithms.rgf2sided     as rgf2sided
import algorithms.hpr_serial    as hprs
import algorithms.hpr_parallel  as hprp
import algorithms.bcr_serial    as bcrs
import algorithms.bcr_parallel  as bcrp
import algorithms.nested_dissection as nd
import algorithms.pdiv_serial   as pdiv_s
import algorithms.pdiv_parallel as pdiv_p
import algorithms.smw           as smw

import verifyResults as verif

import numpy as np
import time

from mpi4py import MPI



if __name__ == "__main__":
    # ---------------------------------------------------------------------------------------------
    # Initialization of the problem and computation of the reference solution
    # ---------------------------------------------------------------------------------------------
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    greenRetardedBenchtiming = {"np": 0, "csc": 0, "rgf": 0, "rgf2sided": 0, "hpr_serial": 0}
    greenLesserBenchtiming   = {"np": 0, "csc": 0, "rgf": 0, "rgf2sided": 0}

    # Problem parameters
    size = 20
    blocksize = 2
    density = blocksize**2/size**2
    bandwidth = np.ceil(blocksize/2)

    isComplex = True
    seed = 63



    # Retarded Green's function initial matrix
    A = genMat.generateBandedDiagonalMatrix(size, bandwidth, isComplex, seed)
    A = transMat.transformToSymmetric(A)
    A_csc = convMat.convertDenseToCSC(A)

    # Retarded Green's function references solutions (Full inversions)
    GreenRetarded_refsol_np, greenRetardedBenchtiming["np"]   = inv.numpyInversion(A)
    GreenRetarded_refsol_csc, greenRetardedBenchtiming["csc"] = inv.scipyCSCInversion(A_csc)

    #vizu.vizualiseDenseMatrixFlat(GreenRetarded_refsol_np, "GreenRetarded_refsol_np")

    if not verif.verifResults(GreenRetarded_refsol_np, GreenRetarded_refsol_csc):
        print("Error: Green retarded references solutions are different.")
        exit()
    else:
        # Extract the blocks from the retarded Green's function reference solution
        GreenRetarded_refsol_block_diag\
        , GreenRetarded_refsol_block_upper\
        , GreenRetarded_refsol_block_lower = convMat.convertDenseToBlocksTriDiagStorage(GreenRetarded_refsol_np, blocksize)

    A_block_diag, A_block_upper, A_block_lower = convMat.convertDenseToBlocksTriDiagStorage(A, blocksize)



    # Lesser Green's function initial matrix
    GreenAdvanced_refsol_np = np.conjugate(np.transpose(GreenRetarded_refsol_np)) 

    SigmaLesser = genMat.generateBandedDiagonalMatrix(size, bandwidth, isComplex, seed)
    SigmaLesser = transMat.transformToSymmetric(SigmaLesser)

    # Lesser Green's function references solutions (Full inversions)
    # 1. Dense matrix
    tic = time.perf_counter()
    B = A @ SigmaLesser @ GreenAdvanced_refsol_np
    toc = time.perf_counter()

    timing = toc - tic

    GreenLesser_refsol_np, greenLesserBenchtiming["np"]   = inv.numpyInversion(B)
    greenLesserBenchtiming["np"] += timing

    

    # 2. CSC matrix
    GreenAdvanced_refsol_csc = convMat.convertDenseToCSC(GreenAdvanced_refsol_np)
    SigmaLesser_csc = convMat.convertDenseToCSC(SigmaLesser)

    tic = time.perf_counter()
    B_csc = A_csc @ SigmaLesser_csc @ GreenAdvanced_refsol_csc
    toc = time.perf_counter()

    timing = toc - tic

    GreenLesser_refsol_csc, greenLesserBenchtiming["csc"] = inv.scipyCSCInversion(B_csc)
    greenLesserBenchtiming["csc"] += timing

    if not verif.verifResults(GreenLesser_refsol_np, GreenLesser_refsol_csc):
        print("Error: Green lesser references solutions are different.")
        exit()
    else:
        # Extract the blocks from the retarded Green's function reference solution
        GreenAdvanced_refsol_block_diag\
        , GreenAdvanced_refsol_block_upper\
        , GreenAdvanced_refsol_block_lower = convMat.convertDenseToBlocksTriDiagStorage(GreenLesser_refsol_np, blocksize)




    # ---------------------------------------------------------------------------------------------
    # 1. RGF  
    # ---------------------------------------------------------------------------------------------

    comm.barrier()
    # .1 Single process RGF
    if rank == 0: # Single process algorithm
        GreenRetarded_rgf_diag\
        , GreenRetarded_rgf_upper\
        , GreenRetarded_rgf_lower\
        , greenRetardedBenchtiming["rgf"] = rgf.rgf_leftToRight_Gr(A_block_diag, A_block_upper, A_block_lower)
        
        #vizu.vizualiseDenseMatrixFromBlocks(GreenRetarded_rgf_diag, GreenRetarded_rgf_upper, GreenRetarded_rgf_lower)

        print("RGF: Gr validation: ", verif.verifResultsBlocksTri(GreenRetarded_refsol_block_diag, 
                                                                 GreenRetarded_refsol_block_upper, 
                                                                 GreenRetarded_refsol_block_lower, 
                                                                 GreenRetarded_rgf_diag, 
                                                                 GreenRetarded_rgf_upper, 
                                                                 GreenRetarded_rgf_lower)) 

    comm.barrier()
    # .1 Double processes RGF
    GreenRetarded_rgf2sided_diag\
    , GreenRetarded_rgf2sided_upper\
    , GreenRetarded_rgf2sided_lower\
    , greenRetardedBenchtiming["rgf2sided"] = rgf2sided.rgf2sided_Gr(A_block_diag, A_block_upper, A_block_lower)

    if rank == 0: # Results agregated on 1st process and compared to reference solution
        print("RGF 2-sided: Gr validation: ", verif.verifResultsBlocksTri(GreenRetarded_refsol_block_diag, 
                                                                          GreenRetarded_refsol_block_upper, 
                                                                          GreenRetarded_refsol_block_lower, 
                                                                          GreenRetarded_rgf2sided_diag, 
                                                                          GreenRetarded_rgf2sided_upper, 
                                                                          GreenRetarded_rgf2sided_lower)) 



    # ---------------------------------------------------------------------------------------------
    # 2. BCR (Block cyclic reduction) 
    # ---------------------------------------------------------------------------------------------

    comm.barrier()
    # .1 Serial BCR
    if rank == 0:
        G_bcr_parallel_inverse = bcrs.inverse_bcr(A, blocksize)

        G_bcr_parallel_inverse_diag  = np.zeros((size, size), dtype=np.complex128)
        G_bcr_parallel_inverse_upper = np.zeros((size, size), dtype=np.complex128)
        G_bcr_parallel_inverse_lower = np.zeros((size, size), dtype=np.complex128)

        G_bcr_parallel_inverse_diag\
        , G_bcr_parallel_inverse_upper\
        , G_bcr_parallel_inverse_lower = convMat.convertDenseToBlocksTriDiagStorage(G_bcr_parallel_inverse, blocksize)

        print("BCR serial: Gr validation: ", verif.verifResultsBlocksTri(GreenRetarded_refsol_block_diag, 
                                                                         GreenRetarded_refsol_block_upper, 
                                                                         GreenRetarded_refsol_block_lower, 
                                                                         G_bcr_parallel_inverse_diag, 
                                                                         G_bcr_parallel_inverse_upper, 
                                                                         G_bcr_parallel_inverse_lower))
    
    comm.barrier()
    # .2 Parallel BCR
    G_bcr_parallel_inverse = bcrp.inverse_bcr(A, blocksize)

    if rank == 0:
        #vizu.vizualiseDenseMatrixFlat(G_bcr_parallel_inverse, "G_bcr_inverse")

        G_bcr_parallel_inverse_diag  = np.zeros((size, size), dtype=np.complex128)
        G_bcr_parallel_inverse_upper = np.zeros((size, size), dtype=np.complex128)
        G_bcr_parallel_inverse_lower = np.zeros((size, size), dtype=np.complex128)

        G_bcr_parallel_inverse_diag\
        , G_bcr_parallel_inverse_upper\
        , G_bcr_parallel_inverse_lower = convMat.convertDenseToBlocksTriDiagStorage(G_bcr_parallel_inverse, blocksize)

        print("BCR parallel: Gr validation: ", verif.verifResultsBlocksTri(GreenRetarded_refsol_block_diag, 
                                                                            GreenRetarded_refsol_block_upper, 
                                                                            GreenRetarded_refsol_block_lower, 
                                                                            G_bcr_parallel_inverse_diag, 
                                                                            G_bcr_parallel_inverse_upper, 
                                                                            G_bcr_parallel_inverse_lower))

    
    # ---------------------------------------------------------------------------------------------
    # 3. HPR (Hybrid Parallel Recurence) 
    # ---------------------------------------------------------------------------------------------

    comm.barrier()
    # .1 Serial HPR
    if rank == 0:
        G_hpr_serial, greenRetardedBenchtiming["hpr_serial"] = hprs.hpr_serial(A, blocksize)

        G_hpr_serial_diag = np.zeros((size, size), dtype=np.complex128)
        G_hpr_serial_upper = np.zeros((size, size), dtype=np.complex128)
        G_hpr_serial_lower = np.zeros((size, size), dtype=np.complex128)

        G_hpr_serial_diag\
        , G_hpr_serial_upper\
        , G_hpr_serial_lower = convMat.convertDenseToBlocksTriDiagStorage(G_hpr_serial, blocksize)

        print("HPR serial: Gr validation: ", verif.verifResultsBlocksTri(GreenRetarded_refsol_block_diag, 
                                                                          GreenRetarded_refsol_block_upper, 
                                                                          GreenRetarded_refsol_block_lower, 
                                                                          G_hpr_serial_diag, 
                                                                          G_hpr_serial_upper, 
                                                                          G_hpr_serial_lower))


    comm.barrier()
    # .2 Parallel HPR
    A_copy = np.copy(A)
    G_hpr_paper = hprp.inverse_hybrid(A_copy, blocksize)
    
    if rank == 0:
        G_hpr_paper_inverse_diag  = np.zeros((size, size), dtype=np.complex128)
        G_hpr_paper_inverse_upper = np.zeros((size, size), dtype=np.complex128)
        G_hpr_paper_inverse_lower = np.zeros((size, size), dtype=np.complex128)

        G_hpr_paper_inverse_diag\
        , G_hpr_paper_inverse_upper\
        , G_hpr_paper_inverse_lower = convMat.convertDenseToBlocksTriDiagStorage(G_hpr_paper, blocksize)

        print("HPR paper: Gr validation: ", verif.verifResultsBlocksTri(GreenRetarded_refsol_block_diag, 
                                                                          GreenRetarded_refsol_block_upper, 
                                                                          GreenRetarded_refsol_block_lower, 
                                                                          G_hpr_paper_inverse_diag, 
                                                                          G_hpr_paper_inverse_upper, 
                                                                          G_hpr_paper_inverse_lower))
        


    # ---------------------------------------------------------------------------------------------
    # 4. Pairwise
    # ---------------------------------------------------------------------------------------------

    comm.barrier()
    # .1 PDIV
    if rank == 0:
        G_pdiv_serial = pdiv_s.pdiv(A, blocksize)

        G_pdiv_serial_diag = np.zeros((size, size), dtype=np.complex128)
        G_pdiv_serial_upper = np.zeros((size, size), dtype=np.complex128)
        G_pdiv_serial_lower = np.zeros((size, size), dtype=np.complex128)

        G_pdiv_serial_diag\
        , G_pdiv_serial_upper\
        , G_pdiv_serial_lower = convMat.convertDenseToBlocksTriDiagStorage(G_pdiv_serial, blocksize)

        print("PDIV serial: Gr validation: ", verif.verifResultsBlocksTri(GreenRetarded_refsol_block_diag, 
                                                                          GreenRetarded_refsol_block_upper, 
                                                                          GreenRetarded_refsol_block_lower, 
                                                                          G_pdiv_serial_diag, 
                                                                          G_pdiv_serial_upper, 
                                                                          G_pdiv_serial_lower))


    comm.barrier()
    # .1 PDIV
    if rank == 0:
        G_pdiv_serial = pdiv_p.pdiv(A, blocksize)

        G_pdiv_serial_diag = np.zeros((size, size), dtype=np.complex128)
        G_pdiv_serial_upper = np.zeros((size, size), dtype=np.complex128)
        G_pdiv_serial_lower = np.zeros((size, size), dtype=np.complex128)

        G_pdiv_serial_diag\
        , G_pdiv_serial_upper\
        , G_pdiv_serial_lower = convMat.convertDenseToBlocksTriDiagStorage(G_pdiv_serial, blocksize)

        print("PDIV parallel: Gr validation: ", verif.verifResultsBlocksTri(GreenRetarded_refsol_block_diag, 
                                                                          GreenRetarded_refsol_block_upper, 
                                                                          GreenRetarded_refsol_block_lower, 
                                                                          G_pdiv_serial_diag, 
                                                                          G_pdiv_serial_upper, 
                                                                          G_pdiv_serial_lower))
    

    # ---------------------------------------------------------------------------------------------
    # X. Data plotting
    # ---------------------------------------------------------------------------------------------
    #if rank == 0:
        #vizu.showBenchmark(greenRetardedBenchtiming, size/blocksize, blocksize, label="Retarded Green's function")

        #vizu.showBenchmark(greenLesserBenchtiming, size/blocksize, blocksize, label="Lesser Green's function")

