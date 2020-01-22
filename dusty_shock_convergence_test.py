import matplotlib.pyplot as plt
from second_order_moving_mesh import *

#Iterate over different cell numbers to test convergence with N
N = 128
Ns = [50, 100,200]#,500]#,1000,2000,5000]
K=1000.0
Ks = [0.1, 1.0, 10.]
t= 5.0
ts = [1e-5, 0.2, 0.6, 1.0, 2.0, 5.0]