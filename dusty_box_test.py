import matplotlib.pyplot as plt
from first_order_moving_mesh import *

xend=1.0
nx=100
dx=xend/nx
x=np.arange(0,nx)*dx
t=5.0

def analytical(pd0, pg0, rho_g0, rho_d0, K, t):		#NB pg0 can be used because gas is constant
	term1 = pd0*np.exp(-K*rho_g0*t) + (rho_d0/rho_g0)*pg0*(1-np.exp(-K*rho_g0*t))


#	Run simulations over a range of timesteps
times = [1e-1, 1, 2, 3, 4, 5]
K=1.0
errs=[]

for t in times:
	#	Set up base grid: initially, velocity of gas is 0, dust velocity initially 1
	fixed = mesh(200, 1.0, mesh_type = "Lagrangian", K =K, CFL = 0.5)
	fixed.setup(vL=0.0, rhoL=1.0, PL=1.0, vLd=1.0, rhoLd=1.0, IC="LRsplit", boundary="flow", cutoff=2.0)
	
	pd0 = fixed.Q[0,4]
	true_pdt = analytical(pd0, 0.0, 1.0, 1.0, K, t)
	
	fixed.solve(tend=t, scheme = "approx", plotsep=100)
	sol_pdt = fixed.Q[0,4]
	
	err = np.abs(pd0 - sol_pdt)
	errs.append(err)

plt.plot(times, errs)
plt.yscale("log")
print(errs)

plt.show()
