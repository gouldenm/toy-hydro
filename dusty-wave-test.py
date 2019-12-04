import matplotlib.pyplot as plt
from dustywave_sol import *
from first_order_moving_mesh import *
#import first-order-moving-mesh

delta = 1e-4 #dv = 10^-4 c_s, drho = 1e-4 rho
gamma = 5./3

#t_s = np.array([0.01, 0.1, 1.0])
#K_list = 1/t_s #since rho_g ~= 1.0

Ks = np.logspace(-4, 4, 10)
vrms=[]

xend = 1.0
nx = 1000
dx = xend/nx
x = np.arange(0, nx)*dx
t=10.0

for K in Ks:
	print(K)
	dw = DustyWaveSolver(delta=delta, K = K, feedback = False)
	sol = dw(t)
	grid = mesh(nx, t, 1.0, K=K, gamma = gamma, mesh_type="Lagrangian")
	grid.setup(drho=delta, drhod=delta, l=1.0)
	grid.solve()
	
	f, ax = plt.subplots(2,1)
	ax[0].plot(x, grid.rho_gas, 'r-')
	ax[0].plot(grid.x, grid.W[:,3], 'r:')
	ax[1].plot(x, grid.v_gas, 'r-', label="gas solver")
	ax[1].plot(grid.x, grid.W[:,4], 'r:', label="dust solver")
	
	ax[0].plot(x, sol.rho_gas(x),  'k-')
	ax[0].plot(x, sol.rho_dust(x), 'k:')
	ax[1].plot(x, sol.v_gas(x),  'k-', label='gas true') 
	ax[1].plot(x, sol.v_dust(x), 'k:', label='dust true')
	plt.pause(1)
	
	rms = np.sqrt(np.mean((grid.v_dust-sol.v_dust(x))**2))
	vrms.append(rms)

print(vrms)
plt.figure()
plt.plot(Ks, vrms)
plt.xscale("log")
plt.title("rms(v_dust) for various stopping times, t=10.0s")
plt.xlabel("K = 1/ts")
plt.ylabel("rms(v_dust)")
plt.show()