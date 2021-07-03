import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats


# Improved errors for fBm with H=0.1 and T=1, geometric scheme.
fBm_M = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
fBm_N = [[1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 91, 128, 181, 256, 362, 512, 724, 1024],
      [1, 2, 3, 4, 5, 8, 11, 16, 22, 32, 45, 64, 90, 128, 181, 256, 362, 512],
      [1, 2, 3, 4, 5, 8, 11, 15, 21, 30, 43, 60, 85, 121, 171, 241, 341],
      [1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 90, 128, 181, 256],
      [1, 2, 3, 5, 6, 9, 13, 18, 26, 36, 51, 72, 102, 145, 205],
      [1, 2, 3, 4, 5, 7, 11, 15, 21, 30, 43, 60, 85, 121, 171],
      [1, 2, 3, 5, 6, 9, 13, 18, 26, 37, 52, 73, 103, 146],
      [1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 90, 128],
      [1, 2, 3, 4, 5, 7, 10, 14, 20, 28, 40, 57, 80, 114],
      [1, 2, 3, 4, 6, 9, 13, 18, 26, 36, 51, 72, 102]]

fBm_errors = [[0.683687, 0.528237, 0.420265, 0.346109, 0.253310, 0.199291, 0.149395, 0.098625, 0.065529, 0.043699, 0.028083, 0.017409, 0.010589, 0.006437, 0.003830, 0.002251, 0.001309, 0.000754, 0.000431, 0.000245],
          [0.635407, 0.451472, 0.333850, 0.256923, 0.204204, 0.119592, 0.076692, 0.039571, 0.022815, 0.010167, 0.004749, 0.002037, 0.000866, 0.000342, 0.000132, 4.91e-05, 1.78e-05, 6.26e-06],
          [0.604933, 0.405954, 0.285941, 0.210481, 0.160789, 0.087336, 0.049924, 0.025489, 0.012368, 0.004540, 0.001559, 0.000543, 0.000158, 4.34e-05, 1.14e-05, 2.86e-06, 6.71e-07],
          [0.582543, 0.374196, 0.253831, 0.180499, 0.103376, 0.069539, 0.035804, 0.013985, 0.005244, 0.001812, 0.000526, 0.000123, 2.74e-05, 5.37e-06, 0., 1.57e-07],
          [0.564594, 0.350288, 0.230415, 0.115370, 0.087745, 0.044988, 0.017360, 0.006881, 0.002035, 0.000596, 0.000130, 2.47e-05, 0., 4.99e-07, 0.],
          [0.549739, 0.331373, 0.212379, 0.143411, 0.101948, 0.048047, 0.021629, 0.008499, 0.003022, 0.000708, 0.000140, 2.59e-05, 3.50e-06, 3.66e-07, 0.],
          [0.537137, 0.315874, 0.197939, 0.091741, 0.068621, 0.031377, 0.010481, 0.003575, 0.000841, 0.000167, 2.69e-05, 3.64e-06, 3.76e-07, 0.],
          [0.526234, 0.302840, 0.186042, 0.120975, 0.062400, 0.035876, 0.014809, 0.004062, 0.001044, 0.000225, 3.50e-05, 3.82e-06, 3.47e-07, 0.],
          [0.516652, 0.291658, 0.176020, 0.112708, 0.077254, 0.043063, 0.017208, 0.005369, 0.001490, 0.000283, 4.10e-05, 0., 0., 0.],
          [0.508122, 0.281912, 0.167430, 0.105750, 0.053486, 0.020847, 0.005956, 0.001722, 0.000307, 5.28e-05, 5.47e-06, 4.40e-07, 2.21e-08]]


def plot_fBm_errors():
    for m in fBm_M:
        n_here = np.array(fBm_N[m-1])
        errors_here = np.array(fBm_errors[m-1])
        plt.loglog(m*n_here[:-1]+1, errors_here[:-1], label=f"m={m}")

    plt.legend(loc="upper right")
    plt.xlabel("Number of nodes")
    plt.ylabel("Error")
    plt.show()