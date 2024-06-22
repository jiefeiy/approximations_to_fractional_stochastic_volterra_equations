import numpy as np
from scipy.special import ndtri
from scipy.stats.qmc import Sobol
import rHestonMarkovSimulation
import RoughKernel


def samples(lambda_, nu, theta, V_0, T, nodes, weights, rho, S_0, r, m, N_time, sample_paths=True, return_times=None,
            qmc=False, rng=None, rv_shift=False):
    """
    Simulates sample paths under the Markovian approximation of the rough Heston model.
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Final time/Time of maturity
    :param N_time: Number of time steps for the simulation
    :param S_0: Initial stock price
    :param r: Interest rate
    :param m: Number of samples
    :param nodes: Nodes of the Markovian approximation
    :param weights: Weights of the Markovian approximation
    :param sample_paths: If True, returns the entire sample paths, not just the final values. Also returns the sample
        paths of the square root of the volatility and the components of the volatility
    :param return_times: Integer that specifies how many time steps are returned. Only relevant if sample_paths is True.
        E.g., N_time is 100 and return_times is 25, then the paths are simulated using 100 equispaced time steps, but
        only the 26 = 25 + 1 values at the times np.linspace(0, T, 26) are returned. May be used especially for storage
        saving reasons, as only these (in this case 26) values are ever stored. The number N_time must be divisible by
        return_times. If return_times is None, it is set to N_time, i.e. we return every time step that was simulated.
    :param qmc: If True, uses Quasi-Monte Carlo simulation with the Sobol sequence. If False, uses standard Monte Carlo
    :param rng: Can specify a sampler to use for sampling the underlying random variables. If qmc is true, expects
        an instance of scipy.stats.qmc.Sobol() with the correctly specified dimension of the simulated random variables.
        If qmc is False, expects an instance of np.random.default_rng()
    :param rv_shift: Only relevant when using QMC. Can specify a shift by which the uniform random variables in [0,1)^d
        are drawn. When the random variables X are drawn from Sobol, instead uses (X + rv_shift) mod 1. If True,
        randomly generates such a random shift
    :return: Numpy array of the simulations, and the rng that was used for generating the underlying random variables
    """
    if sample_paths is False:
        return_times = 1
    if return_times is None:
        return_times = N_time
    if N_time % return_times != 0:
        raise ValueError(f'The number of time steps for the simulation N_time={N_time} is not divisible by the number'
                         f'of time steps that should be returned return_times={return_times}.')
    saving_steps = N_time // return_times
    dt = T / N_time
    N = len(nodes)
    if N == 1:
        nodes = np.array([nodes[0], 2 * nodes[0] + 1])
        weights = np.array([weights[0], 0])
        N = 2
        one_node = True
    else:
        one_node = False

    if rng is None:
        if qmc:
            rng = Sobol(d=2 * N_time, scramble=False)
        else:
            rng = np.random.default_rng()
    if isinstance(rv_shift, bool) and rv_shift:
        dim = 2 * N_time
        rv_shift = np.random.uniform(0, 1, dim)

    """
    if qmc:
        if int(2 ** np.ceil(np.log2(m)) + 0.001) != m_input:
            print(f'Using QMC requires simulating a number m of samples that is a power of 2. The input m={m_input} '
                  f'is not a power of 2.')
    """

    V_init = V_0 / nodes / (np.sum(weights / nodes))

    A = np.eye(N) + np.diag(nodes) * dt + lambda_ * weights[None, :] * dt
    A_inv = np.linalg.inv(A)
    b = theta * dt + (nodes * V_init)[:, None] * dt

    def step_SV(log_S_, V_comp_, dBW_):
        sq_V = np.sqrt(np.fmax(weights @ V_comp_, 0))
        log_S_ = log_S_ + r * dt + sq_V * (rho * dBW_[:, 1] + np.sqrt(1 - rho ** 2) * dBW_[:, 0]) \
            - 0.5 * sq_V ** 2 * dt
        V_comp_ = A_inv @ (V_comp_ + nu * (sq_V * dBW_[:, 1])[None, :] + b)
        return log_S_, V_comp_

    def generate_samples():
        if qmc:
            rv = rng.random(n=m)
            if isinstance(rv_shift, np.ndarray):
                rv = (rv + rv_shift) % 1.
            if rv[0, 0] == 0.:
                rv[1:, :] = np.sqrt(dt) * ndtri(rv[1:, :])  # first sample is 0 and cannot be inverted
                rv[0, :] = 0.
            else:
                rv = np.sqrt(dt) * ndtri(rv)

            return lambda index: rv[:, index::N_time]
        else:
            return lambda index: np.sqrt(dt) * rng.standard_normal((m, 2))

    result = np.empty((N + 2, return_times + 1, m)) if sample_paths else np.empty((N + 2, m))

    dBW = generate_samples()
    current_V_comp = np.empty((N, m))
    current_V_comp[:, :] = V_init[:, None]
    current_log_S = np.full(m, np.log(S_0))
    for i in range(N_time):
        current_log_S, current_V_comp = step_SV(current_log_S, current_V_comp, dBW(i))

        if sample_paths and (i + 1) % saving_steps == 0:
            result[0, (i + 1) // saving_steps] = current_log_S
            result[2:, (i + 1) // saving_steps] = current_V_comp

    if sample_paths:
        result[0, 0, :] = np.log(S_0)
        result[2:, 0, :] = V_init[:, None]
        result[1, :, :] = np.fmax(np.einsum('i,ijk->jk', weights, result[2:, :, :]), 0)
    else:
        result[1, :] = np.fmax(np.einsum('i,ij->j', weights, result[2:, :]), 0)

    # result = result[..., :m]
    # result[0, ...] = np.exp(result[0, ...])
    if one_node:
        result = result[:-1, ...]

    return result, dBW


if __name__ == '__main__':
    H = 0.1
    T = 1
    N = 1
    rate = 0.06
    time_horizon = 1

    nodes, weights = RoughKernel.quadrature_rule(H, N, T)
    # result, dBW = samples(lambda_=0.3, nu=0.3, theta=0.02,
    #                             V_0=0.02, T=time_horizon, nodes=nodes, weights=weights,
    #                             rho=-0.7, S_0=100, r=rate, m=400000, N_time=100)
    result2, _ = rHestonMarkovSimulation.samples(lambda_=0.3, nu=0.3, theta=0.02,
                          V_0=0.02, T=time_horizon, nodes=nodes, weights=weights,
                          rho=-0.7, S_0=100, r=rate, m=10000, N_time=100)

    print(nodes)
    print(weights)
    # print(result.shape)
    # print(dBW(49)[0:5, 1])

    euro_put_price = np.exp(-rate*time_horizon) * np.mean(np.maximum(105 - (result2[0, -1, :]), 0.0))
    print(euro_put_price)

