import matplotlib.pyplot as plt
from dustywave_sol import *
from second_order_moving_mesh import *

delta = 1e-4
gamma = 5./3

#v_rms = []



#Iterate over different cell numbers to test convergence with N
N = 100
Ns = [50, 100,200]#,500]#,1000,2000,5000]
K=1.0
Ks = [0.01, 0.1, 1.0, 10., 100.]
t= 5.0
ts = [1e-5, 0.2, 0.6, 1.0, 2.0, 5.0]


"""grid = mesh(N, 1.0, K=1., gamma=gamma, mesh_type="Fixed")
grid.setup(drho=delta, drhod=delta, l=1.0, IC="soundwave", boundary="periodic")
grid.solve(tend=t, scheme="exp", feedback=True, plotsep=200)
"""

for N in Ns:
	xend = 1.0
	nx = N
	dx = xend/nx
	x = np.arange(0, nx)*dx
	dw = DustyWaveSolver(delta=delta, K = K, feedback = True)
	sol = dw(t)
	grid = mesh(N, 1.0, K=K, gamma=gamma, mesh_type="Fixed")
	grid.setup(drho=delta, drhod=delta, l=1.0, IC="soundwave", boundary="periodic")
	grid.solve(tend=t, scheme="exp", feedback=True)
	
	grid2 = mesh(N, 1.0, K=K, gamma=gamma, mesh_type="Fixed")
	grid2.setup(drho=delta, drhod=delta, l=1.0, IC="soundwave", boundary="periodic")
	grid2.solve(tend=t, scheme="exp", feedback=True, order2=True)
	
	f, ax = plt.subplots(2,1)
	ax[1].set_ylabel(r"\$v$")
	ax[0].set_title("N=" + str(N) + ", t=" + str(t) + ", K=" + str(K))
	ax[0].set_ylabel(r"$\rho$")
	
	ax[0].plot(grid.pos, grid.rho_gas, 'k-', label=r'gas $\rho$ 1st')
	ax[0].plot(grid.pos, grid.rho_dust, 'r-', label=r'dust $\rho$ 1st')
	
	ax[0].plot(grid2.pos, grid2.rho_gas, 'k-', alpha=0.5, label=r'gas $\rho$ 2nd')
	ax[0].plot(grid2.pos, grid2.rho_dust, 'b-', alpha=0.5, label=r'dust $\rho$ 2nd')
	
	ax[1].plot(grid.pos, grid.v_dust, 'r-', label=r"dust $v$ 1st")
	ax[1].plot(grid.pos, grid.v_gas, 'k-', label=r"gas $v$ 1st")
	
	ax[1].plot(grid2.pos, grid2.v_dust, 'b-', alpha=0.5, label=r"dust $v$ 2nd")
	ax[1].plot(grid2.pos, grid2.v_gas, 'k-', alpha=0.5, label=r"gas $v$ 2nd")
	
	ax[0].plot(x, sol.rho_gas(x),  'k:', label=r'gas $\rho$ true')
	ax[0].plot(x, sol.rho_dust(x), 'r:', label=r'dust $\rho$ true')
	
	ax[1].plot(x, sol.v_gas(x),  'k:', label=r'gas $v$ true') 
	ax[1].plot(x, sol.v_dust(x), 'r:', label=r'dust $v$ true')
	
	ax[0].legend()
	
	ax[1].legend()
	plt.pause(0.1)

plt.show()