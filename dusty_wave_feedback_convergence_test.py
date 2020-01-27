import matplotlib.pyplot as plt
from dustywave_sol import *
from test_second_order import *
#from second_order_moving_mesh import *

delta = 1e-4
gamma = 5./3

#



#Iterate over different cell numbers to test convergence with N
N = 128
Ns = np.array([128, 258, 512, 1000])
K=0.1
Ks = [0.1, 1.0, 10.]
t= 3.0
ts = [1e-5, 0.2, 0.6, 1.0, 2.0, 5.0]

"""
grid = mesh(N, 1.0, K=1., gamma=gamma, mesh_type="Fixed")
grid.setup(drho=delta, drhod=delta, l=1.0, IC="soundwave", boundary="periodic")
grid.solve(tend=0.5, scheme="exp", feedback=False, order2=True, plotsep=40)

plt.show()
"""
rms1 = []
rms2 = []

for N in Ns:
	xend = 1.0
	nx = N
	dx = xend/nx
	x = np.arange(0, nx)*dx
	dw = DustyWaveSolver(delta=delta, K = K, feedback = False)
	sol = dw(t)	
	grid = mesh(N, 1.0, K=K, gamma=gamma, mesh_type="Fixed")
	grid.setup(drho=delta, drhod=delta, l=1.0, IC="soundwave", boundary="periodic")
	grid.solve(tend=t, scheme="exp", feedback=False)
	
	grid2 = mesh(N, 1.0, K=K, gamma=gamma, mesh_type="Fixed")
	grid2.setup(drho=delta, drhod=delta, l=1.0, IC="soundwave", boundary="periodic")
	grid2.solve(tend=t, scheme="exp", feedback=False, order2=True)
	
	f, ax = plt.subplots(2,2)#, sharey="row")
	
	f.suptitle("N=" + str(N) + ", t=" + str(t) + ", K=" + str(K))
	ax[0,0].set_ylabel(r"$\rho$")
	ax[1,0].set_ylabel(r"$v$")
	
	ax[0,0].plot(grid.pos, grid.rho_gas, 'r-', label=r'gas $\rho$ 1st')
	ax[0,1].plot(grid.pos, grid.rho_dust, 'r-', label=r'dust $\rho$ 1st')
	
	ax[0,0].plot(grid2.pos, grid2.rho_gas, 'b-', alpha=0.5, label=r'gas $\rho$ 2nd')
	ax[0,1].plot(grid2.pos, grid2.rho_dust, 'b-', alpha=0.5, label=r'dust $\rho$ 2nd')
	
	ax[1,0].plot(grid.pos, grid.v_gas, 'r-', label=r"gas $v$ 1st")
	ax[1,1].plot(grid.pos, grid.v_dust, 'r-', label=r"dust $v$ 1st")
	
	ax[1,0].plot(grid2.pos, grid2.v_gas, 'b-', alpha=0.5, label=r"gas $v$ 2nd")
	ax[1,1].plot(grid2.pos, grid2.v_dust, 'b-', alpha=0.5, label=r"dust $v$ 2nd")
	
	ax[0,0].plot(x, sol.rho_gas(x),  'k:', label=r'gas $\rho$ true')
	ax[0,1].plot(x, sol.rho_dust(x), 'k:', label=r'dust $\rho$ true')
	
	ax[1,0].plot(x, sol.v_gas(x),  'k:', label=r'gas $v$ true') 
	ax[1,1].plot(x, sol.v_dust(x), 'k:', label=r'dust $v$ true')
	
	ax[0,0].legend()
	ax[1,0].legend()
	ax[1,1].legend()
	ax[0,1].legend()
	
	rms_1 = np.sqrt(np.mean((grid.v_gas-sol.v_gas(x))**2))
	rms_2 = np.sqrt(np.mean((grid2.v_gas-sol.v_gas(x))**2))
	
	rms1.append(rms_1)
	rms2.append(rms_2)
	
	plt.pause(1.0)

plt.figure()
plt.xscale("log")
plt.yscale("log")
plt.plot(Ns, rms1, label="First order")
plt.plot(Ns, rms2, label="Second order")
plt.plot(Ns, 1./Ns**2, label="1/N^2")
plt.plot(Ns, 1./Ns, label="1/N")
plt.title("Delta = " + str(delta))
plt.legend()
plt.ylabel("rms(v_gas)")
plt.xlabel("N")

plt.show()