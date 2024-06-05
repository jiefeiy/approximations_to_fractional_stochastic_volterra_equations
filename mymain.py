import rHestonMarkovSimulation
import RoughKernel

H = -0.2
T = 1
N = 2

nodes, weights = RoughKernel.quadrature_rule(H, N, T)
rHestonMarkovSimulation.samples(lambda_=0.3, nu=0.3, theta=0.02,
                                V_0=0.02, T=1, nodes=nodes, weights=weights,
                                rho=-0.7, S_0=1, r=0.05, m=1024, N_time=50)

print(nodes)
print(weights)

