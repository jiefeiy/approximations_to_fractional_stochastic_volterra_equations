import numpy as np
import scipy.interpolate, scipy.special
import ComputationalFinance as cf
import scipy.stats
from scipy.special import ndtri
from scipy.stats.qmc import Sobol
from numpy.random import default_rng
import functions
from os.path import exists
import psutil


def samples(lambda_, nu, theta, V_0, T, nodes, weights, rho, S_0, r, m, N_time, sample_paths=False, return_times=None,
            vol_only=False, euler=False, qmc=True, sampler=None, rv_shift=False, verbose=0):
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
    :param vol_only: If True, simulates only the volatility process, not the stock price process
    :param euler: If True, uses an Euler scheme. If False, uses moment matching
    :param qmc: If True, uses Quasi-Monte Carlo simulation with the Sobol sequence. If False, uses standard Monte Carlo
    :param sampler: Can specify a sampler to use for sampling the underlying random variables. If qmc is true, expects
        an instance of scipy.stats.qmc.Sobol() with the correctly specified dimension of the simulated random variables.
        If qmc is False, expects an instance of np.random.default_rng()
    :param rv_shift: Only relevant when using QMC. Can specify a shift by which the uniform random variables in [0,1)^d
        are drawn. When the random variables X are drawn from Sobol, instead uses (X + rv_shift) mod 1. If True,
        randomly generates such a random shift
    :param verbose: Determines the number of intermediary results printed to the console
    :return: Numpy array of the simulations, and the sampler that was used for sampling the underlying random variables
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

    if sampler is None:
        if qmc:
            if vol_only:
                sampler = Sobol(d=N_time, scramble=False)
            else:
                if euler:
                    sampler = Sobol(d=2 * N_time, scramble=False)
                else:
                    sampler = Sobol(d=3 * N_time, scramble=False)
        else:
            sampler = np.random.default_rng()
    if isinstance(rv_shift, bool) and rv_shift:
        dim = N_time
        if not vol_only and euler:
            dim = 2 * N_time
        elif not vol_only and not euler:
            dim = 3 * N_time
        rv_shift = np.random.uniform(0, 1, dim)


    m_input = m  # the original input of how many samples should be simulated
    m_return = m  # the final number of samples that we will actually return
    # m itself is the number of samples that we simulate
    # We always have m >= m_return >= m_input

    if qmc:
        m = int(2 ** np.ceil(np.log2(m)) + 0.001)
        m_return = m
        if m != m_input:
            print(f'Using QMC requires simulating a number m of samples that is a power of 2. The input m={m_input} '
                  f'is not a power of 2. Simulates m={m} samples instead.')

    available_memory = np.sqrt(psutil.virtual_memory().available)
    necessary_memory = 2.5 * np.sqrt(N + 2) * np.sqrt(return_times + 1) * np.sqrt(m) * np.sqrt(np.array([0.]).nbytes)
    if necessary_memory > available_memory:
        raise MemoryError(f'Not enough memory to store the sample paths of the rough Heston model with'
                          f'{N} Markovian dimensions, {return_times} time points where the sample paths should be '
                          f'returned and {m} sample paths. Roughly {necessary_memory}**2 bytes needed, '
                          f'while only {available_memory}**2 bytes are available.')

    available_memory_for_random_variables = available_memory / 3
    necessary_memory_for_random_variables = np.sqrt(3 * N_time) * np.sqrt(m) * np.sqrt(np.array([0.]).nbytes)
    number_rounds = int(np.ceil(necessary_memory_for_random_variables / available_memory_for_random_variables))
    m_per_round = int(np.ceil(m_return / number_rounds))
    m = m_per_round * number_rounds

    V_init = np.zeros(N)
    V_init[0] = V_0 / weights[0]

    if euler:
        A = np.eye(N) + np.diag(nodes) * dt + lambda_ * weights[None, :] * dt
        A_inv = np.linalg.inv(A)
        b = theta * dt + (nodes * V_init)[:, None] * dt

        if vol_only:
            def step_SV(V_comp_, dW_):
                sq_V = np.sqrt(np.fmax(weights @ V_comp_, 0))
                # dW_samples = cf.rand_normal(loc=0, scale=np.sqrt(dt), size=m, antithetic=antithetic)
                return A_inv @ (V_comp_ + nu * (sq_V * dW_)[None, :] + b)
        else:
            def step_SV(log_S_, V_comp_, dBW_):
                sq_V = np.sqrt(np.fmax(weights @ V_comp_, 0))
                # dW = cf.rand_normal(loc=0, scale=np.sqrt(dt), size=m, antithetic=antithetic)
                # dB = cf.rand_normal(loc=0, scale=np.sqrt(dt), size=m, antithetic=antithetic)
                log_S_ = log_S_ + r * dt + sq_V * (rho * dBW_[:, 1] + np.sqrt(1 - rho ** 2) * dBW_[:, 0]) \
                    - 0.5 * sq_V ** 2 * dt
                V_comp_ = A_inv @ (V_comp_ + nu * (sq_V * dBW_[:, 1])[None, :] + b)
                return log_S_, V_comp_

    else:
        weight_sum = np.sum(weights)
        A = -(np.diag(nodes) + lambda_ * weights[None, :]) * dt / 2
        exp_A = scipy.linalg.expm(A)
        b = (nodes * V_init + theta) * dt / 2
        ODE_b = np.linalg.solve(A, (exp_A - np.eye(N)) @ b)[:, None]
        z = weight_sum ** 2 * nu ** 2 * dt
        rho_bar_sq = 1 - rho ** 2
        rho_bar = np.sqrt(rho_bar_sq)

        def ODE_step_V(V_):
            return exp_A @ V_ + ODE_b

        B = (6 + np.sqrt(3)) / 4
        A = B - 0.75

        def SDE_step_V(V_, dW_):
            x = weights @ V_
            # dW_ = cf.rand_uniform(size=m, antithetic=antithetic)
            temp = np.sqrt((3 * z) * x + (B * z) ** 2)
            p_1 = (z / 2) * x * ((A * B - A - B + 1.5) * z + (np.sqrt(3) - 1) / 4 * temp + x) / (
                    (x + B * z - temp) * temp * (temp - (B - A) * z))
            p_2 = x / (1.5 * x + A * (B - A / 2) * z)
            test_1 = dW_ < p_1
            test_2 = p_1 + p_2 <= dW_
            x_step = A * z * np.ones(len(temp))
            x_step[test_1] = B * z - temp[test_1]
            x_step[test_2] = B * z + temp[test_2]
            return V_ + (x_step / weight_sum)[None, :]

        def step_V(V_, dW_):
            return ODE_step_V(SDE_step_V(ODE_step_V(V_), dW_))

        def SDE_step_B(log_S_, V_, dB_):
            # dB_ = cf.rand_normal(loc=0, scale=np.sqrt(dt / 2), size=m, antithetic=antithetic)
            x = weights @ V_
            return log_S_ + np.sqrt(x) * rho_bar * dB_ - (0.5 * rho_bar_sq * dt / 2) * x, V_

        drift_SDE_step_W = - (nodes[0] * V_init[0] + theta) * dt
        fact_1 = dt / 2 * (lambda_ - 0.5 * rho * nu)

        def SDE_step_W(log_S_, V_, dW_):
            V_new = step_V(V_, dW_)
            dY = V_ + V_new
            log_S_new = log_S_ + r * dt + rho / nu * (drift_SDE_step_W + (dt / 2 * nodes[0]) * dY[0, :]
                                                      + fact_1 * (weights @ dY) + (V_new[0, :] - V_[0, :]))
            return log_S_new, V_new

        if vol_only:
            def step_SV(V_, dW_):
                return step_V(V_, dW_)
        else:
            def step_SV(S_, V_, dBW_):
                return SDE_step_B(*SDE_step_W(*SDE_step_B(S_, V_, dBW_[:, 0]), dBW_[:, 2]), dBW_[:, 1])

    def generate_samples():
        if vol_only:
            if qmc:
                rv = sampler.random(n=m_per_round)
                if isinstance(rv_shift, np.ndarray):
                    rv = (rv + rv_shift) % 1.
                if euler:
                    if rv[0, 0] == 0.:
                        rv[1:, :] = np.sqrt(dt) * ndtri(rv[1:, :])  # first sample is 0 and cannot be inverted
                        rv[0, :] = 0.
                    else:
                        rv = np.sqrt(dt) * ndtri(rv)
            else:
                if euler:
                    rv = np.sqrt(dt) * sampler.standard_normal((m_per_round, N_time))
                else:
                    rv = sampler.uniform(0, 1, (m_per_round, N_time))
        else:
            if qmc:
                rv = sampler.random(n=m_per_round)
                if isinstance(rv_shift, np.ndarray):
                    rv = (rv + rv_shift) % 1.
                if euler:
                    if rv[0, 0] == 0.:
                        rv[1:, :] = np.sqrt(dt) * ndtri(rv[1:, :])  # first sample is 0 and cannot be inverted
                        rv[0, :] = 0.
                    else:
                        rv = np.sqrt(dt) * ndtri(rv)
                else:
                    if rv[0, 0] == 0.:  # first sample is 0 and cannot be inverted
                        rv[1:, :2 * N_time] = np.sqrt(dt / 2) * ndtri(rv[1:, :2 * N_time])
                        rv[0, :2 * N_time] = 0.
                    else:
                        rv[:, :2 * N_time] = np.sqrt(dt / 2) * ndtri(rv[:, :2 * N_time])
            else:
                if euler:
                    rv = np.sqrt(dt) * sampler.standard_normal((m_per_round, 2 * N_time))
                else:
                    rv = np.empty((m_per_round, 3 * N_time))
                    rv[:, :2 * N_time] = np.sqrt(dt / 2) * sampler.standard_normal((m_per_round, 2 * N_time))
                    rv[:, 2 * N_time:] = sampler.uniform(0, 1, (m_per_round, N_time))
        return rv

    if vol_only:
        result = np.empty((N + 1, m, return_times + 1)) if sample_paths else np.empty((N + 1, m))
        for j in range(number_rounds):
            dW = generate_samples()
            current_V_comp = np.empty((N, m_per_round))
            current_V_comp[:, :] = V_init[:, None]
            for i in range(N_time):
                if verbose >= 1:
                    print(f'Simulation round {j + 1} of {number_rounds}, step {i + 1} of {N_time}')
                current_V_comp = step_SV(current_V_comp, dW[:, i])
                if sample_paths and (i + 1) % saving_steps == 0:
                    result[1:, j * m_per_round:(j + 1) * m_per_round, (i + 1) // saving_steps] = current_V_comp
            if not sample_paths:
                result[1:, j * m_per_round:(j + 1) * m_per_round] = current_V_comp

        if sample_paths:
            result[1:, :, 0] = V_init[:, None]
            result[0, :, :] = np.fmax(np.einsum('i,ijk->jk', weights, result[1:, :, :]), 0)
        else:
            result[0, :] = np.fmax(np.einsum('i,ij->j', weights, result[1:, :]), 0)

    else:
        result = np.empty((N + 2, m, return_times + 1)) if sample_paths else np.empty((N + 2, m))
        for j in range(number_rounds):
            dBW = generate_samples()
            current_V_comp = np.empty((N, m_per_round))
            current_V_comp[:, :] = V_init[:, None]
            current_log_S = np.full(m_per_round, np.log(S_0))
            for i in range(N_time):
                if verbose >= 1:
                    print(f'Simulation round {j + 1} of {number_rounds}, step {i + 1} of {N_time}')
                current_log_S, current_V_comp = step_SV(current_log_S, current_V_comp, dBW[:, i::N_time])
                if sample_paths and (i + 1) % saving_steps == 0:
                    result[0, j * m_per_round:(j + 1) * m_per_round, (i + 1) // saving_steps] = current_log_S
                    result[2:, j * m_per_round:(j + 1) * m_per_round, (i + 1) // saving_steps] = current_V_comp
            if not sample_paths:
                result[0, j * m_per_round:(j + 1) * m_per_round] = current_log_S
                result[2:, j * m_per_round:(j + 1) * m_per_round] = current_V_comp
        if sample_paths:
            result[0, :, 0] = np.log(S_0)
            result[2:, :, 0] = V_init[:, None]
            result[1, :, :] = np.fmax(np.einsum('i,ijk->jk', weights, result[2:, :, :]), 0)
        else:
            result[1, :] = np.fmax(np.einsum('i,ij->j', weights, result[2:, :]), 0)

    result[0, ...] = np.exp(result[0, ...])
    if one_node:
        result = result[:-1, ...]
    return result[:, :m_return, ...], sampler


def eur(K, lambda_, rho, nu, theta, V_0, S_0, T, nodes, weights, r, m, N_time, euler=False, qmc=True, payoff='call',
        implied_vol=False):
    """
    Gives the price or the implied volatility of a European option in the approximated, Markovian rough Heston model
    using MC or QMC simulation.
    :param K: Strike prices, assumed to be a numpy array
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param S_0: Initial stock price
    :param T: Final time/Time of maturity
    :param nodes: The nodes of the Markovian approximation
    :param weights: The weights of the Markovian approximation
    :param r: Interest rate
    :param m: Number of samples
    :param N_time: Number of time steps used in simulation
    :param euler: If True, uses an Euler scheme. If False, uses moment matching
    :param qmc: If True, uses Quasi-Monte Carlo simulation with the Sobol sequence. If False, uses standard Monte Carlo
    :param payoff: The payoff function, or the string 'call' or the string 'put'
    :param implied_vol: If True (only for payoff 'call' or 'put') returns the implied volatility, else returns the price
    return: The prices of the call option for the various strike prices in K
    """
    samples_, _ = samples(lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, m=m, S_0=S_0, r=r, N_time=N_time,
                          nodes=nodes, weights=weights, sample_paths=False, euler=euler, qmc=qmc)[0, :]
    return cf.eur_MC(S_0=S_0, K=K, T=T, r=r, samples=samples_, payoff=payoff, implied_vol=implied_vol)


def price_geom_asian_call(K, lambda_, rho, nu, theta, V_0, S_0, T, nodes, weights, r, m, N_time, euler=False, qmc=True):
    """
    Gives the price of a European call option in the approximated, Markovian rough Heston model.
    :param K: Strike prices, assumed to be a numpy array
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param S_0: Initial stock price
    :param T: Final time/Time of maturity
    :param nodes: The nodes of the Markovian approximation
    :param weights: The weights of the Markovian approximation
    :param r: Interest rate
    :param m: Number of samples
    :param N_time: Number of time steps used in simulation
    :param euler: If True, uses an Euler scheme. If False, uses moment matching
    :param qmc: If True, uses Quasi-Monte Carlo simulation with the Sobol sequence. If False, uses standard Monte Carlo
    return: The prices of the call option for the various strike prices in K
    """
    samples_, _ = samples(lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, m=m, S_0=S_0, N_time=N_time,
                          nodes=nodes, weights=weights, r=r, sample_paths=True, euler=euler, qmc=qmc)[0, :, :]
    return cf.price_geom_asian_call_MC(K=K, samples=samples_)


def price_avg_vol_call(K, lambda_, nu, theta, V_0, T, nodes, weights, m, N_time, euler=False, qmc=True):
    """
    Gives the price of a European call option in the approximated, Markovian rough Heston model.
    :param K: Strike prices, assumed to be a numpy array
    :param lambda_: Mean-reversion speed
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Final time/Time of maturity
    :param nodes: The nodes of the Markovian approximation
    :param weights: The weights of the Markovian approximation
    :param m: Number of samples
    :param N_time: Number of time steps used in simulation
    :param euler: If True, uses an Euler scheme. If False, uses moment matching
    :param qmc: If True, uses Quasi-Monte Carlo simulation with the Sobol sequence. If False, uses standard Monte Carlo
    return: The prices of the call option for the various strike prices in K
    """
    samples_, _ = samples(lambda_=lambda_, nu=nu, theta=theta, V_0=V_0, T=T, m=m, N_time=N_time, nodes=nodes,
                          weights=weights, r=0., S_0=1., rho=0., sample_paths=True, vol_only=True, euler=euler,
                          qmc=qmc)[0, :, :]
    return cf.price_avg_vol_call_MC(K=K, samples=samples_)


def am_features(x, degree=6, K=0.):
    """
    Computes the features for the pricing of American options under Markovian approximations of rough Heston using the
    Longstaff-Schwartz algorithm.
    :param x: Array of samples of the Markov process. Is of shape (N + 1, m), where N is the dimension of the Markovian
        approximation and m is the number of samples. The array x[0, :] is the array of stock price samples, the array
        x[1:, :] is the array of the N-dimensions Markovian approximation of the volatility process, already multiplied
        with their weights (i.e. w_i V^i_t)
    :param degree: Maximal weighted degree of features polynomials. The degree of the stock price carries weight 1,
        the degree of the total volatility carries weight 2, and the degree of the components of the volatility carries
        weight 3. The last (highest mean-reversion) component of the volatility is not used
    :param K: Strike price. Rough size of the stock price for normalization
    return: The feature vector, of shape (m, dim), where dim is the number of features
    """
    m = x.shape[-1]
    d = x.shape[0]
    normalized_stock = ((x[0, :] - K) / K) if np.abs(K) > 0.01 else x[0, :]
    vol = np.sum(x[1:, :], axis=0)
    vol_factors = x[1:-1, :].T
    if degree == 1:
        dim = 1
    elif degree == 2:
        dim = 3
    else:
        dim = d + 3
    feat = np.empty((m, dim))
    if degree >= 1:
        feat[:, 0] = normalized_stock
    if degree >= 2:
        feat[:, 1] = normalized_stock ** 2
        feat[:, 2] = vol
    if degree >= 3:
        feat[:, 3:5] = normalized_stock[:, None] * feat[:, 1:3]
        feat[:, 5:d + 3] = vol_factors
    current_N = 4
    current_ind = d + 3
    lower_N_stock = 3
    upper_N_stock = d + 3
    lower_N_vol = 2
    upper_N_vol = 3
    next_lower_N_vol = 5
    next_upper_N_vol = d + 3
    lower_N_vol_factors = np.arange(5, d + 3, dtype=int)
    upper_N_vol_factors = d + 3
    while current_N <= degree:
        feat_new = np.empty((m, feat.shape[1] + upper_N_stock - lower_N_stock + upper_N_vol - lower_N_vol
                             + (current_N % 3 == 0)
                             * (upper_N_vol_factors * (d - 2) - np.sum(lower_N_vol_factors, dtype=int))))
        # print(feat_new.shape[1])
        feat_new[:, :current_ind] = feat
        feat = feat_new
        next_ind = current_ind + upper_N_stock - lower_N_stock
        feat[:, current_ind:next_ind] = normalized_stock[:, None] * feat[:, lower_N_stock:upper_N_stock]
        lower_N_stock = current_ind
        current_ind = next_ind
        next_ind = current_ind + upper_N_vol - lower_N_vol
        feat[:, current_ind:next_ind] = vol[:, None] * feat[:, lower_N_vol:upper_N_vol]
        lower_N_vol = next_lower_N_vol
        next_lower_N_vol = current_ind
        current_ind = next_ind
        if current_N % 3 == 0:
            for i in range(d - 2):
                next_ind = current_ind + upper_N_vol_factors - lower_N_vol_factors[i]
                feat[:, current_ind:next_ind] = \
                    vol_factors[:, i:i + 1] * feat[:, lower_N_vol_factors[i]:upper_N_vol_factors]
                lower_N_vol_factors[i] = current_ind
                current_ind = next_ind
            upper_N_vol_factors = current_ind
        upper_N_stock = current_ind
        upper_N_vol = next_upper_N_vol
        next_upper_N_vol = current_ind
        current_N = current_N + 1
    return feat


def price_am(K, lambda_, rho, nu, theta, V_0, S_0, T, nodes, weights, payoff, r, m, N_time, N_dates, feature_degree=6,
             euler=False, qmc=True, qmc_error_estimators=25, verbose=0):
    """
    Gives the price of an American option in the approximated, Markovian rough Heston model.
    :param K: Strike price
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param S_0: Initial stock price
    :param T: Final time/Time of maturity
    :param nodes: The nodes of the Markovian approximation
    :param weights: The weights of the Markovian approximation
    :param payoff: The payoff function, either 'call' or 'put' or a function taking as inputs S (samples) and K (strike)
    :param r: Interest rate
    :param m: Number of samples. Uses m for fitting the stopping rule, and another m for pricing
    :param N_time: Number of time steps used in the simulation
    :param N_dates: Number of exercise dates. If None, N_dates = N_time
    :param feature_degree: The degree of the polynomial features used in the regression
    :param euler: If True, uses an Euler scheme. If False, uses moment matching
    :param qmc: If True, uses Quasi-Monte Carlo simulation with the Sobol sequence. If False, uses standard Monte Carlo
    :param qmc_error_estimators: Runs the pricing step of the Longstaff-Schwartz algorithm qmc_error_estimators times
        to get an MC estimate for the QMC error
    :param verbose:
    return: The prices of the call option for the various strike prices in K
    """
    if payoff == 'call':
        def payoff(S):
            return cf.payoff_call(S=S, K=K)
    elif payoff == 'put':
        def payoff(S):
            return cf.payoff_put(S=S, K=K)
    if N_dates is None:
        N_dates = N_time
    if qmc:
        m_ = int(2 ** np.ceil(np.log2(m)) + 0.001)
        if m != m_:
            print(f'Using QMC requires simulating a number m of samples that is a power of 2. The input m={m} '
                  f'is not a power of 2. Simulates m={m_} samples instead.')
            m = m_

    def features(x):
        return am_features(x=x, degree=feature_degree, K=K)

    '''
    kind = 'sample paths'
    params = {'S': 1., 'K': np.array([K]), 'H': 0.1, 'T': T, 'lambda': lambda_, 'rho': rho, 'nu': nu, 'theta': theta, 'V_0': V_0,
              'rel_tol': 0., 'r': r}
    vol_simulation = 'euler' if euler else 'mackevicius'
    filename = functions.get_filename(N=len(nodes), mode='BL2', euler=euler, antithetic=antithetic, N_time=N_time,
                                      kind=kind, params=params, truth=False, markov=False)
    filename = f'rHeston sample paths {len(nodes)} dim BL2 {vol_simulation} antithetic {N_time} time steps, H=0.1, ' \
                   f'lambda={lambda_:.3}, rho={rho:.3}, nu={nu:.3}, theta={theta:.3}, r = 0.06, V_0={V_0:.3}, ' \
                   f'T=1.0.npy'
    if exists(filename):  # delete this in any published version. It is trash
        samples_orig = np.load(filename)
    else:
        samples_orig = samples(lambda_=lambda_, nu=nu, theta=theta, V_0=V_0, T=T, nodes=nodes, weights=weights, rho=rho,
                               S_0=S_0, r=r, m=m, N_time=N_time, sample_paths=True, return_times=ex_times, vol_only=False,
                               euler=euler, antithetic=antithetic)
    '''
    def get_samples(sampler_=None, rv_shift=False):
        samples_, sampler_ = samples(lambda_=lambda_, nu=nu, theta=theta, V_0=V_0, T=T, nodes=nodes, weights=weights,
                                     rho=rho, S_0=S_0, r=r, m=m, N_time=N_time, sample_paths=True, return_times=N_dates,
                                     vol_only=False, euler=euler, qmc=qmc, sampler=sampler_, rv_shift=rv_shift)
        samples_[1:-1, :, :] = weights[:, None, None] * samples_[2:, :, :]
        samples_ = samples_[:-1, :, :]
        samples_[1:, :, :] = samples_[1:, :, :] - samples_[1:, :, :1]
        return samples_, sampler_

    samples_1, sampler = get_samples()
    (biased_est, biased_stat), models = cf.price_am(T=T, r=r, samples=samples_1, payoff=payoff, features=features)
    if qmc:
        estimates = np.empty(qmc_error_estimators)
        for i in range(qmc_error_estimators):
            samples_1, sampler = get_samples(sampler, False if i == 0 else True)
            sampler.reset()
            sampler.fast_forward(n=m)
            estimates[i], _ = cf.price_am_forward(T=T, r=r, samples=samples_1, payoff=payoff, models=models,
                                                  features=features)
            print(estimates[i])
        est, stat = cf.MC(estimates)
    else:
        samples_1, sampler = get_samples(sampler)
        est, stat = cf.price_am_forward(T=T, r=r, samples=samples_1, payoff=payoff, models=models, features=features)
    return est, stat, biased_est, biased_stat, models, features
