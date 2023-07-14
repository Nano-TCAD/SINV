"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-07

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import numpy as np

def nested_dissection_1width_separator_fillin(matsize: int, height: int) -> np.array:
    """
        Compute the fillin of the matrix for each level of a nested dissection
        algorithm using a 1-width separator.
    """

    level_fillin   = np.zeros(height, dtype=int)
    previous_level = np.zeros(height, dtype=int)

    # Initialisation of the recursive formula
    level_fillin[0] = matsize * matsize

    domain_size = np.sqrt(matsize)

    nelem_S = 1 * domain_size
    nelem_L = (domain_size - 1)/2 * domain_size
    nelem_R = (domain_size - 1)/2 * domain_size

    SS = nelem_S * nelem_S
    SL = nelem_S * nelem_L
    SR = nelem_S * nelem_R
    LL = nelem_L * nelem_L
    LR = nelem_L * nelem_R

    previous_level[1] = SS + 2*SL + 2*SR
    level_fillin[1]   = previous_level[1] + LL + LR
    
    for i in range(2, height, 1):

        nelem_S = (nelem_S-1)/2
        nelem_L = (nelem_L-nelem_S)/2
        nelem_R = (nelem_R-nelem_S)/2

        SS = nelem_S * nelem_S
        SL = nelem_S * nelem_L
        SR = nelem_S * nelem_R
        previous_level[i] = previous_level[i-1] + pow(2, i-1) * (SS + 2*SL + 2*SR)

        LL = nelem_L * nelem_L
        LR = nelem_L * nelem_R
        level_fillin[i] = previous_level[i] + pow(2, i-1) * (LL + LR)

    return level_fillin


if __name__ == '__main__':
    matsize = 1000
    height  = 8
    number_of_nonzero_elems = nested_dissection_1width_separator_fillin(matsize, height)
    pourcentage_fillin = [number_of_nonzero_elems[i]/(matsize*matsize) for i in range(height)]

    import matplotlib.pyplot as plt
    # Plot the two curves
    fig, ax = plt.subplots()
    ax.plot(pourcentage_fillin, label='1-width separator')
    ax.set_xlabel('Level')
    ax.set_ylabel('Fill-in [%]')
    ax.set_title('Fill-in of the matrix for each level of a 1-width nested dissection algorithm')
    ax.legend()
    plt.show()