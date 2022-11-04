import numpy as np
from scipy.special import gamma
import rHestonBackbone
import psutil


def adams_scheme(F, H, T, N_Riccati):
    """
    Applies the Adams scheme to solve h(t) = int_0^t K(t - s) F(s, h(s)) ds.
    :param H: Hurst parameter
    :param F: Right-hand side (time-dependent!)
    :param T: Final time
    :param N_Riccati: Number of time steps used for solving the fractional Riccati equation
    :return: The solution h
    """
    dim = len(F(0, 0))
    available_memory = np.sqrt(psutil.virtual_memory().available)
    necessary_memory = np.sqrt(5 * dim) * np.sqrt(N_Riccati) * np.sqrt(np.array([0.], dtype=np.cdouble).nbytes)
    if necessary_memory > available_memory:
        raise MemoryError(f'Not enough memory to compute the characteristic function of the rough Heston model with'
                          f'{dim} inputs and {N_Riccati} time steps. Roughly {necessary_memory}**2 bytes needed, '
                          f'while only {available_memory}**2 bytes are available.')

    dt = T / N_Riccati
    coefficient = dt ** (H + 0.5) / gamma(H + 2.5)
    v_1 = np.arange(N_Riccati + 1) ** (H + 1.5)
    v_2 = np.arange(N_Riccati + 1) ** (H + 0.5)
    v_3 = coefficient * (v_1[N_Riccati:1:-1] + v_1[N_Riccati - 2::-1] - 2 * v_1[N_Riccati - 1:0:-1])
    v_4 = coefficient * (v_1[:-1] - (np.arange(N_Riccati) - H - 0.5) * v_2[1:])
    v_5 = dt ** (H + 0.5) / gamma(H + 1.5) * (v_2[N_Riccati:0:-1] - v_2[N_Riccati - 1::-1])

    h = np.zeros(shape=(dim, N_Riccati + 1), dtype=np.cdouble)
    F_vec = np.zeros(shape=(dim, N_Riccati), dtype=np.cdouble)
    F_vec[:, 0] = F(0, h[:, 0])
    h[:, 1] = F_vec[:, 0] * v_4[0] + coefficient * F(0, F_vec[:, 0] * v_5[-1])
    for k in range(2, N_Riccati + 1):
        F_vec[:, k - 1] = F((k - 1) / N_Riccati, h[:, k - 1])
        h[:, k] = F_vec[:, 0] * v_4[k - 1] + F_vec[:, 1:k] @ v_3[-k + 1:] \
            + coefficient * F(k / N_Riccati, F_vec[:, :k] @ v_5[-k:])

    return h


def cf_log_price(z, S_0, H, lambda_, rho, nu, theta, V_0, T, N_Riccati=200):
    """
    Gives the characteristic function of the log-price in the rough Heston model.
    :param z: Argument of the characteristic function (assumed to be a numpy array)
    :param S_0: Initial stock price
    :param H: Hurst parameter
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Final time
    :param N_Riccati: Number of time steps used for solving the fractional Riccati equation
    :return: The characteristic function
    """
    z = complex(0, 1) * z
    a = nu * nu / 2
    b = rho * nu * z - lambda_
    c = (z - 1) * z / 2

    def F(t, x):
        return c + (b + a * x) * x

    psi = adams_scheme(F=F, H=H, T=T, N_Riccati=N_Riccati)
    integral = np.trapz(psi, dx=T / N_Riccati)
    integral_sq = np.trapz(psi ** 2, dx=T / N_Riccati)

    return np.exp(z * np.log(S_0) + V_0 * T * c + (theta + V_0 * b) * integral + V_0 * a * integral_sq)


def cf_avg_log_price(z, S_0, H, lambda_, rho, nu, theta, V_0, T, N_Riccati=200):
    """
    Gives the characteristic function of the average (on [0, T]) log-price in the rough Heston model.
    :param z: Argument of the characteristic function (assumed to be a numpy array)
    :param S_0: Initial stock price
    :param H: Hurst parameter
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Final time
    :param N_Riccati: Number of time steps used for solving the fractional Riccati equation
    :return: The characteristic function
    """
    z = complex(0, 1) * z
    dt = T / N_Riccati
    a_sq = z ** 2

    def F(t, x):
        return (0.5 * t ** 2) * a_sq - (0.5 * t) * z + ((rho * nu * t) * z - lambda_ + (nu ** 2 / 2) * x) * x

    h = adams_scheme(F=F, H=H, T=T, N_Riccati=N_Riccati)

    integral = np.trapz(h, dx=dt)
    integral_squared = np.trapz(h ** 2, dx=dt)
    integral_time = np.trapz(h * np.linspace(0, 1, N_Riccati + 1), dx=dt)
    return np.exp(z * np.log(S_0) + T * V_0 * (z / 6 - 0.25) * z + (theta - lambda_ * V_0) * integral
                  + nu ** 2 * V_0 / 2 * integral_squared + nu * rho * V_0 * integral_time * z)


def cf_avg_vol(z, H, lambda_, nu, theta, V_0, T, N_Riccati=200):
    """
    Gives the characteristic function of the average (on [0, T]) volatility in the rough Heston model.
    :param z: Argument of the characteristic function (assumed to be a numpy array)
    :param H: Hurst parameter
    :param lambda_: Mean-reversion speed
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Final time
    :param N_Riccati: Number of time steps used for solving the fractional Riccati equation
    :return: The characteristic function
    """
    z = complex(0, 1) * z
    dt = T / N_Riccati

    def F(t, x):
        return z / T + (-lambda_ + (nu ** 2 / 2) * x) * x

    h = adams_scheme(F=F, H=H, T=T, N_Riccati=N_Riccati)

    integral = np.trapz(h, dx=dt)
    integral_squared = np.trapz(h ** 2, dx=dt)
    return np.exp(z * V_0 + (theta - lambda_ * V_0) * integral + nu ** 2 * V_0 / 2 * integral_squared)


def iv_eur_call(S_0, K, H, lambda_, rho, nu, theta, V_0, T, rel_tol=1e-03, verbose=0):
    """
    Gives the implied volatility of the European call option in the rough Heston model. Uses Fourier inversion.
    :param S_0: Initial stock price
    :param K: Strike price, assumed to be a numpy array
    :param H: Hurst parameter
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Maturity
    :param rel_tol: Required maximal relative error in the implied volatility
    :param verbose: Determines how many intermediate results are printed to the console
    return: The implied volatility of the call option
    """
    return rHestonBackbone.call(char_fun=lambda u, T_, N_: cf_log_price(u, S_0, H, lambda_, rho, nu, theta, V_0, T_,
                                                                        N_),
                                S_0=S_0, K=K, T=T, rel_tol=rel_tol, verbose=verbose, option='european', output='iv')


def skew_eur_call(S_0, H, lambda_, rho, nu, theta, V_0, T, rel_tol=1e-03, verbose=0):
    """
    Gives the skew of the European call option in the rough Heston model. Uses Fourier inversion.
    :param S_0: Initial stock price
    :param H: Hurst parameter
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Maturity
    :param rel_tol: Required maximal relative error in the implied volatility
    :param verbose: Determines how many intermediate results are printed to the console
    return: The skew of the call option
    """
    return rHestonBackbone.skew_eur_call(char_fun=lambda u, T_, N_: cf_log_price(u, S_0, H, lambda_, rho, nu, theta,
                                                                                 V_0, T_, N_),
                                         T=T, rel_tol=rel_tol, verbose=verbose)


def price_geom_asian_call(S_0, K, H, lambda_, rho, nu, theta, V_0, T, rel_tol=1e-03, verbose=0):
    """
    Gives the price of the geometric Asian call option in the rough Heston model. Uses Fourier inversion.
    :param S_0: Initial stock price
    :param K: Strike price, assumed to be a numpy array
    :param H: Hurst parameter
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Maturity
    :param rel_tol: Required maximal relative error in the implied volatility
    :param verbose: Determines how many intermediate results are printed to the console
    return: The implied volatility of the call option
    """
    return rHestonBackbone.call(char_fun=lambda u, T_, N_: cf_avg_log_price(u, S_0, H, lambda_, rho, nu, theta, V_0, T_,
                                                                            N_),
                                S_0=S_0, K=K, T=T, rel_tol=rel_tol, verbose=verbose, option='geometric asian',
                                output='price')


def price_avg_vol_call(K, H, lambda_, nu, theta, V_0, T, rel_tol=1e-03, verbose=0):
    """
    Gives the price of the average volatility European call option in the rough Heston model. Uses Fourier inversion.
    :param K: Strike price, assumed to be a numpy array
    :param H: Hurst parameter
    :param lambda_: Mean-reversion speed
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Maturity
    :param rel_tol: Required maximal relative error in the implied volatility
    :param verbose: Determines how many intermediate results are printed to the console
    return: The implied volatility of the call option
    """
    return rHestonBackbone.call(char_fun=lambda u, T_, N_: cf_avg_vol(u, H, lambda_, nu, theta, V_0, T_, N_),
                                S_0=V_0, K=K, T=T, rel_tol=rel_tol, verbose=verbose, option='average volatility',
                                output='price')
