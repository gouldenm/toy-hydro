import matplotlib.pyplot as plt
from dustywave_sol import *
from first_order_moving_mesh import *
#import first-order-moving-mesh

delta = 1e-4 #dv = 10^-4 c_s, drho = 1e-4 rho
gamma = 5./3

#t_s = np.array([0.01, 0.1, 1.0])
#K_list = 1/t_s #since rho_g ~= 1.0

Ks = np.array([ 0.01, 1.0, 100.])
vrms_exp=[]
vrms_approx=[]

xend = 1.0
nx = 200
dx = xend/nx
x = np.arange(0, nx)*dx
t=5.0


for K in Ks:
	dw = DustyWaveSolver(delta=delta, K = K, feedback = False)
	sol = dw(t)
	grid = mesh(nx, 1.0, K=K, gamma = gamma, mesh_type="Lagrangian")
	grid.setup(drho=delta, drhod=delta, l=1.0)
	grid.solve(tend = t, scheme="approx")
	
	"""grida = mesh(nx, 1.0, K=K, gamma = gamma, mesh_type="Lagrangian")
	grida.setup(drho=delta, drhod=delta, l=1.0)
	grida.solve(tend = t, scheme="approx")"""
	
	f, ax = plt.subplots(2,1)
	ax[0].set_title("K=" + str(K))
	ax[0].set_ylabel(r"$\rho$")
	ax[0].plot(x, grid.rho_gas, 'r-', label=r'gas $\rho$ calc')
	ax[0].plot(grid.x, grid.W[:,3], 'r:', label=r'dust $\rho$ calc')
	ax[1].set_ylabel(r"\$v$")
	ax[1].plot(grid.x, grid.W[:,4], 'r:', label=r"dust $v$ calc")
	ax[1].plot(grid.x, grid.W[:,1], 'r-', label=r"gas $v$ calc")
	
	"""ax[0].plot(x, grida.rho_gas, 'b-')
	ax[0].plot(grida.x, grida.W[:,3], 'b:')
	ax[1].plot(x, grida.v_gas, 'b-', label="gas")
	ax[1].plot(grida.x, grida.W[:,4], 'b:', label="dust approx")"""
	
	ax[0].plot(x, sol.rho_gas(x),  'k-', label=r'gas $\rho$ true')
	ax[0].plot(x, sol.rho_dust(x), 'k:', label=r'gas $\rho$ true')
	ax[0].legend()
	ax[1].plot(x, sol.v_gas(x),  'k-', label=r'gas $v$ true') 
	ax[1].plot(x, sol.v_dust(x), 'k:', label=r'dust $v$ true')
	ax[1].legend()
	plt.pause(0.1)
	
	rms_approx = np.sqrt(np.mean((grid.v_dust-sol.v_dust(x))**2)) / np.abs(grid.v_dust)
	vrms_approx.append(rms_approx)
	
	"""rms_approx = np.sqrt(np.mean((grida.v_dust-sol.v_dust(x))**2)) / np.abs(grida.v_dust)
	vrms_approx.append(rms_approx)"""
"""

#print(vrms_exp)
print(vrms_approx)
plt.figure()
plt.plot(Ks, vrms_approx, label="Approx")
#plt.plot(Ks, vrms_exp, label="Exp")
plt.xscale("log")
plt.title("rms(v_dust) for various stopping times, t=10.0s")
plt.xlabel("K = 1/ts")
plt.ylabel("rms(v_dust)")
plt.legend()"""
plt.show()
