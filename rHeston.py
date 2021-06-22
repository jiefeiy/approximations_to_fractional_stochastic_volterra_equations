import time
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import linalg
from scipy.special import gamma
import ComputationalFinance as cf


def sqrt_cov_matrix_rHeston(H=0.1, T=1., N=1000):
    """
    Computes the Cholesky decomposition of the covariance matrix for the rough Heston model with Hurst parameter H,
    final time T and N number of time steps. This is the Cholesky decomposition of the covariance matrix of the
    Gaussian vector (int_0^(T/N) (T/N - s)^(H-1/2) dW_s, ..., int_0^(T/N) (T-s)^(H-1/2) dW_s, W_(T/N)).
    :param H: Hurst parameter
    :param T: Final time
    :param N: Number of time steps
    :return: The Cholesky decomposition of the covariance matrix
    """
    dt = T / N
    cov_matrix = np.empty(shape=(N + 1, N + 1))
    cov_matrix[:-1, :-1] = dt ** (2 * H) / (H + 0.5) * np.array([[(np.fmax(i, j) - 0.5) ** (H - 0.5) * (
            np.fmin(i, j) ** (H + 0.5) - (np.fmin(i, j) - 1) ** (H + 0.5)) for i in range(1, N + 1)] for j in
                                                                 range(1, N + 1)])
    for i in range(N):
        cov_matrix[i, i] = dt ** (2 * H) / (2 * H) * ((i + 1) ** (2 * H) - i ** (2 * H))
    cov_matrix[:-1, -1] = 1 / (H + 0.5) * dt ** (H + 0.5) * np.array(
        [((i + 1) ** (H + 0.5) - i ** (H + 0.5)) for i in range(N)])
    cov_matrix[-1, :-1] = cov_matrix[:-1, -1]
    cov_matrix[-1, -1] = dt
    (L, D, P) = scipy.linalg.ldl(cov_matrix)
    return np.dot(L, np.sqrt(np.fmax(D, np.zeros(shape=D.shape))))


def plot_rHeston(H=0.1, T=1., N=1000, rho=-0.9, lambda_=0.3, theta=0.02, nu=0.3, V_0=0.02, S_0=1.):
    """
    Plots a realization of the rough Heston model together with the variance process.
    :param H: Hurst parameter
    :param T: Final time
    :param N: Number of time steps
    :param rho: Correlation between the Brownian motion driving the volatility and the Brownian motion driving the stock
    :param lambda_: Mean-reversion speed
    :param theta: Mean volatility
    :param nu: Volatility of volatility
    :param V_0: Initial volatility
    :param S_0: Initial stock price
    :return: Nothing, plots a realization of the rough Heston model together with the variance process.
    """
    dt = T / N
    sqrt_cov = sqrt_cov_matrix_rHeston(H, T, N)
    gaussian_increments = sqrt_cov.dot(np.random.normal(0, 1, size=(N + 1, N)))
    # CHECK IF THE RESULT IS AN (N+1, N) ARRAY!!! If so, i is for (t_i-s)^(H-1/2) and j is for int_t_(j-1)^t_j
    V = np.zeros(shape=(N + 1,))
    V[0] = V_0
    coefficients = lambda_ / (H + 0.5) * dt ** (H + 0.5) * np.arange(N, -1, -1) ** (H + 0.5)
    for i in range(1, N + 1):
        V[i] = V_0 + 1 / gamma(H + 0.5) * np.sum(
            (theta - V[:i]) * (coefficients[-i - 1:-1] - coefficients[-i:])
            + nu * np.sqrt(V[:i]) * gaussian_increments[i - 1, :i])
        V[i] = np.fmax(V[i], 0)
    S = np.zeros(shape=(N + 1,))
    S[0] = S_0
    W_2_diff = np.random.normal(0, np.sqrt(dt), size=N)
    for i in range(N):
        S[i + 1] = S[i] * np.exp(
            np.sqrt(V[i]) * (rho * gaussian_increments[-1, i] + np.sqrt(1 - rho ** 2) * W_2_diff[i]) - V[i] * dt / 2)
    times = np.arange(N + 1) * dt
    plt.plot(times, S)
    plt.plot(times, V)
    plt.show()


def rHeston(H=0.1, T=1., N=1000, rho=-0.9, lambda_=0.3, theta=0.02, nu=0.3, V_0=0.02, S_0=1., m=1000, rounds=1):
    """
    Computes m final stock values of the rough Heston model.
    :param H: Hurst parameter
    :param T: Final time
    :param N: Number of time steps
    :param rho: Correlation between the Brownian motion driving the volatility and the Brownian motion driving the stock
    :param lambda_: Mean-reversion speed
    :param theta: Mean volatility
    :param nu: Volatility of volatility
    :param V_0: Initial volatility
    :param S_0: Initial stock price
    :param m: Number of stock prices computed
    :param rounds: Actually generates m*rounds samples, but only m at a time. This is to avoid excessive memory usage.
    :return: An array of all the final stock prices
    """
    dt = T / N
    sqrt_cov = sqrt_cov_matrix_rHeston(H, T, N)
    S = np.empty(shape=(m * rounds))
    for rd in range(rounds):
        gaussian_increments = np.array([sqrt_cov.dot(np.random.normal(0, 1, size=(N + 1, N))) for _ in range(m)])
        V = np.zeros(shape=(m, N + 1))
        V[:, 0] = V_0 * np.ones(m)
        coefficients = lambda_ / (H + 0.5) * dt ** (H + 0.5) * np.arange(N, -1, -1) ** (H + 0.5)
        for i in range(1, N + 1):
            V[:, i] = V[:, 0] + 1 / gamma(H + 0.5) * ((theta - V[:, :i]).dot(coefficients[-i-1:-1] - coefficients[-i:])
                            + np.sum(nu * np.sqrt(V[:, :i]) * gaussian_increments[:, i - 1, :i], axis=1))
            V[:, i] = np.fmax(V[:, i], 0)
        S_ = np.zeros(shape=(m, N + 1,))
        S_[:, 0] = S_0 * np.ones(m)
        W_2_diff = np.random.normal(0, np.sqrt(dt), size=(m, N))
        for i in range(N):
            S_[:, i + 1] = S_[:, i] * np.exp(
                np.sqrt(V[:, i]) * (rho * gaussian_increments[:, -1, i] + np.sqrt(1 - rho ** 2) * W_2_diff[:, i]) -
                V[:, i] * dt / 2)
        S[rd * m:(rd + 1) * m] = S_[:, -1]
    return S


def implied_volatility_call_rHeston(H=0.1, T=1., N=1000, rho=-0.9, lambda_=0.3, theta=0.02, nu=0.3, V_0=0.02, S_0=1.,
                                    K=1., m=1000, rounds=1):
    """
    Computes the implied volatility of a European option under the rough Heston model.
    :param H: Hurst parameter
    :param T: Final time
    :param N: Number of time steps
    :param rho: Correlation between the Brownian motion driving the volatility and the Brownian motion driving the stock
    :param lambda_: Mean-reversion speed
    :param theta: Mean volatility
    :param nu: Volatility of volatility
    :param V_0: Initial volatility
    :param S_0: Initial stock price
    :param K: Strike price
    :param m: Number of stock prices computed
    :param rounds: Actually uses m*rounds samples, but only m at a time to avoid excessive memory usage
    :return: An array of all the final stock prices
    """
    tic = time.perf_counter()
    samples = rHeston(H, T, N, rho, lambda_, theta, nu, V_0, S_0, m, rounds)
    toc = time.perf_counter()
    print(f"Generating {m * rounds} rHeston samples with N={N} takes {np.round(toc - tic, 2)} seconds.")
    return cf.volatility_smile_call(samples, K, T, S_0)