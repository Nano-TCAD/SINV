"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-06

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import numpy as np

from matplotlib import pyplot as plt


class Benchmark:
    # Public methods and attributes
    simulationName : str
    n_runs         : int

    def __init__(self, simulationName : str, n_runs : int):
        self.simulationName = simulationName
        self.n_runs         = n_runs

    def addMethodBenchmark(self, methodName : str, timings : list):
        """
            Add a benchmark for a method.
        """
        self.__methodsTimings[methodName] = timings

    def getMeans(self):
        methodsMeans   : dict = {}

        return methodsMeans
    
    def getStdDeviations(self):
        methodsStdDevs : dict = {}

        return methodsStdDevs


    # Private methods and attributes
    __methodsTimings : dict = {}

    def getMethodMean(self, methodName : str):
        methodMeans : dict = {}

        

        for i in range(self.__methodsTimings[methodName]):


        for key, value in self.__methodsTimings[methodName].items():
            methodMeans[key] = np.mean(value)

        print(f"methodMeans={methodMeans}")

        nAlgsParts = len(self.__methodsTimings[methodName])

        print(f"nAlgsParts={nAlgsParts}")
        print(self.__methodsTimings[methodName])

        return 0
    
    def getMethodStdDeviation(self, methodName : str):
        return np.std(self.__methodsTimings[methodName])





class BenchTiming:
    def __init__(self, simulationName, methodName, nRuns):
        self.simulationName = simulationName
        self.methodName     = methodName
        self.nRuns          = nRuns
        self.timingRuns     = np.zeros(nRuns)

    def getMean(self):
        return np.mean(self.timingRuns)
    
    def getStdDeviation(self):
        return np.std(self.timingRuns)
    
    simulationName : str = ""
    methodName     : str = ""
    nRuns          : int
    timingRuns     : np.ndarray
    


def showBenchmark(benchmarks: list, nBlocks, blockSize):
    """
        Show a bar plot of the benchmark.
    """

    nBenchmarks = len(benchmarks)
    means       = np.zeros(nBenchmarks)
    labels      = ["" for i in range(nBenchmarks)]
    stdDevs     = np.zeros(nBenchmarks)

    for i in range(nBenchmarks):
        means[i]   = benchmarks[i].getMean()
        labels[i]  = benchmarks[i].methodName
        stdDevs[i] = benchmarks[i].getStdDeviation()

    fig, ax = plt.subplots()
    ax.bar(range(nBenchmarks), means, yerr=stdDevs, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel("Time (s)")
    ax.set_xticks(range(nBenchmarks), labels)
    ax.set_title(benchmarks[0].simulationName)
    #ax.suptitle(f"matrixSize={(int)(nBlocks*blockSize)}, nBlocks={(int)(nBlocks)}, blockSize={blockSize}")
    ax.yaxis.grid(True)

    plt.show()