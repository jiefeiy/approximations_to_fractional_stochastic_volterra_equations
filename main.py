import time
import matplotlib.pyplot as plt
import numpy as np
import Data
import rBergomi
import rHeston
from functions import *
import rBergomiMarkov
import rHestonMomentMatching
from scipy.stats import norm
import scipy.special

'''
BS_paths = cf.BS_paths(sigma=0.2, T=1., m=100000, n=1000)
print(1)
geom_avg = np.exp(np.trapz(np.log(BS_paths), dx=1/1000, axis=-1))
print(2)
payoff = np.array(np.average(cf.payoff_call(S=geom_avg, K=1.)))
print(3)
iv = cf.iv_geom_asian_call(S_0=1., K=1., T=1., price=payoff)
print(iv)
time.sleep(36000)
'''
iv = rHeston.price_geom_asian_call(S_0=1., K=np.exp(np.linspace(-1., 0.5, 61)), H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1., rel_tol=1e-03, verbose=10)
plt.plot(np.linspace(-1., 0.5, 61), iv)
plt.show()
print('Finished')
time.sleep(360000)

for i in range(1, 11):
    print(rk.quadrature_rule(H=0.1, T=1., N=i, mode='european'))
print('Finished')
time.sleep(36000)

for N_times in 2 ** np.arange(9):
    '''
    tic = time.perf_counter()
    compute_final_rHeston_stock_prices(params='simple', Ns=np.array([2]), N_times=N_times, modes=['european'], vol_behaviours=['sticky'], recompute=True, m=20000000)
    print(time.perf_counter() - tic)
    tic = time.perf_counter()
    compute_final_rHeston_stock_prices(params='simple', Ns=np.array([2]), N_times=N_times, modes=['european'], vol_behaviours=['mackevicius random'], recompute=True, m=20000000)
    print(time.perf_counter() - tic)
    tic = time.perf_counter()
    compute_final_rHeston_stock_prices(params='simple', Ns=np.array([2]), N_times=N_times, modes=['european'], vol_behaviours=['mackevicius sequential'], recompute=True, m=20000000)
    print(time.perf_counter() - tic)'''

    params = {'T': 0.004, 'K': np.exp(np.linspace(-0.4, 0.4, 33))}
    tic = time.perf_counter()
    compute_final_rHeston_stock_prices(params=rHeston_params(params), Ns=np.array([1]), N_times=N_times, modes=['european'], vol_behaviours=['mackevicius sequential antithetic'], recompute=False, m=1000000, sample_paths=True)
    print(time.perf_counter() - tic)

print('Finished')
# time.sleep(360000)

# print(rk.quadrature_rule(0.1, 2, 1))
k = np.sqrt(0.004) * np.linspace(-1.5, 0.75, 451)[220:-70:5]# [280:-140:10]# [220:-70:5]
params = {'K': np.exp(k), 'T': 0.004}
params = rHeston_params(params)
# true_smile = Data.true_iv_surface_eur_call[-1, 220:-70:5]
print(k, len(k))

# simulation_errors_depending_on_node_size(params=params, verbose=1, true_smile=true_smile, N_times=2**np.arange(4, 10), largest_nodes=np.linspace(0, 10, 101)/0.04, vol_behaviour='sticky')
# optimize_kernel_approximation_for_simulation_vector_inputs(Ns=np.array([1]), N_times=2 ** np.arange(6, 9), params=params, true_smile=true_smile, plot=True, recompute=True, vol_behaviours=['hyperplane reset'], m=10000000)

# compute_strong_discretization_errors(Ns=np.array([2]), N_times=2 ** np.arange(14), N_time_ref=2 ** 14, vol_behaviours=['hyperplane reset', 'ninomiya victoir', 'sticky'], plot=True)
# compute_smiles_given_stock_prices(params=params, Ns=np.array([2]), N_times=2 ** np.arange(16), modes=None, vol_behaviours=['hyperplane reset', 'ninomiya victoir', 'sticky'], plot=True, true_smile=true_smile)
compute_smiles_given_stock_prices(params=params, Ns=np.array([1, 2, 3]), N_times=2 ** np.arange(9), modes=['european'], vol_behaviours=['mackevicius sequential antithetic'], plot=True, true_smile=None)
# compute_smiles_given_stock_prices(params=params, Ns=np.array([2]), N_times=2 ** np.arange(10), modes=['european'], vol_behaviours=['ninomiya victoir', 'correct ninomiya victoir', 'sticky'], plot='vol_behaviour', true_smile=true_smile)
# compute_smiles_given_stock_prices(params=params, Ns=np.array([2]), N_times=2 ** np.arange(10), modes=['european', 'fitted'], vol_behaviours=['hyperplane reset'], plot='mode', true_smile=true_smile)
print('Finished')

