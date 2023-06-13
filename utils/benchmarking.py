"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-06

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import numpy as np

from matplotlib import pyplot as plt



def showBenchmark(benchmarks: list, nBlocks, blockSize, label):
    """
        Show a stacked bar plot of the benchmark.
    """
    nBenchmarks = len(benchmarks)

    Methods = (
    "Numpy",
    "Scipy",
    "RGF",
    "RGF 2-Sided",
    "HPR Serial"
    )

    algsDecompositionTimings = {}
    for benchmark in benchmarks:
        for key, val in benchmark.items():
            algsDecompositionTimings[key] = []
    
    for benchmark in benchmarks:
        for key in algsDecompositionTimings.keys():
            if key not in benchmark.keys():
                algsDecompositionTimings[key].append(0)
            else:
                algsDecompositionTimings[key].append(benchmark[key])

    width = 0.5

    fig, ax = plt.subplots()
    bottom = np.zeros(nBenchmarks)

    for boolean, weight_count in algsDecompositionTimings.items():
        p = ax.bar(Methods, weight_count, width, label=boolean, bottom=bottom)
        bottom += weight_count

    ax.set_title(label)
    ax.legend(loc="upper right")

    plt.show()