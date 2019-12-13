import matplotlib.pyplot as plt
from first_order_moving_mesh import *

xend=1.0
nx=100
dx=xend/nx
x=np.arange(0,nx)*dx
t=5.0

def analytical(pd0, pg0, rho_g0, rho_d0, K, t):		#NB pg0 can be used because gas is constant
	sol = pd0*np.exp(-K*rho_g0*t) + (rho_d0/rho_g0)*pg0*(1-np.exp(-K*rho_g0*t))
	return(sol)


#	Run simulations over a range of timesteps
times = [0.1, 0.3, 0.5, 0.8, 1, 2, 3]
K=10
errs_exp=[]
errs_bw=[]

for t in times:
	#	Set up base grid: initially, velocity of gas is 0, dust velocity initially 1
	backwards = mesh(200, 1.0, mesh_type = "Lagrangian", K =K, CFL = 0.5)
	backwards.setup(vL=0.0, rhoL=1.0, PL=1.0, vLd=1.0, rhoLd=1.0, IC="LRsplit", boundary="flow", cutoff=2.0)
	
	pd0 = backwards.Q[0,4]
	true_pdt = analytical(pd0, 0.0, 1.0, 1.0, K, t)
	
	backwards.solve(tend=t, scheme = "approx")
	bw_pdt = backwards.Q[1,4]
	
	exp = mesh(200, 1.0, mesh_type = "Lagrangian", K =K, CFL = 0.5)
	exp.setup(vL=0.0, rhoL=1.0, PL=1.0, vLd=1.0, rhoLd=1.0, IC="LRsplit", boundary="flow", cutoff=2.0)
	exp.solve(tend=t, scheme = "exp")
	exp_pdt = exp.Q[1,4]
	
	err_exp = np.abs(true_pdt - exp_pdt)
	errs_exp.append(err_exp)
	
	err_bw = np.abs(true_pdt - bw_pdt)
	errs_bw.append(err_bw)

plt.figure()
plt.plot(times, errs_bw, "b-", label="BW error")
plt.plot(times, errs_exp, "r--", label="Exp error")
plt.yscale("log")
plt.xscale("log")
plt.xlabel("time")
plt.ylabel("absolute error")
plt.title("Error convergence with time")
plt.legend()
plt.pause(1)



Ns = [10, 20, 40, 50, 100, 200, 300, 400, 500]
K=10
t=1.0
errs_exp=[]
errs_bw=[]

for N in Ns:
	#	Set up base grid: initially, velocity of gas is 0, dust velocity initially 1
	backwards = mesh(N, 1.0, mesh_type = "Lagrangian", K =K, CFL = 0.5)
	backwards.setup(vL=0.0, rhoL=1.0, PL=1.0, vLd=1.0, rhoLd=1.0, IC="LRsplit", boundary="flow", cutoff=2.0)
	
	pd0 = backwards.Q[0,4]
	true_pdt = analytical(pd0, 0.0, 1.0, 1.0, K, t)
	
	backwards.solve(tend=t, scheme = "approx")
	bw_pdt = backwards.Q[1,4]
	
	exp = mesh(N, 1.0, mesh_type = "Lagrangian", K =K, CFL = 0.5)
	exp.setup(vL=0.0, rhoL=1.0, PL=1.0, vLd=1.0, rhoLd=1.0, IC="LRsplit", boundary="flow", cutoff=2.0)
	exp.solve(tend=t, scheme = "exp")
	exp_pdt = exp.Q[1,4]
	
	err_exp = np.abs(true_pdt - exp_pdt)
	errs_exp.append(err_exp)
	
	err_bw = np.abs(true_pdt - bw_pdt)
	errs_bw.append(err_bw)

plt.figure()
plt.plot(Ns, errs_bw, "b-", label="BW error")
plt.plot(Ns, errs_exp, "r--", label="Exp error")
plt.plot(Ns, 1e-4/np.array(Ns), 'k', label='1/N')
plt.plot(Ns, 1e-3/np.array(Ns)**2, '0.5', label='1/N^2')
plt.yscale("log")
plt.xscale("log")
plt.xlabel("N")
plt.ylabel("absolute error")
plt.title("Error convergence with N")
plt.legend()

plt.show()
