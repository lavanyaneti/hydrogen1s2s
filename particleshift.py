import scipy.integrate as integrate
import scipy.special as special
import numpy as np
from scipy.integrate import quad
from numpy import sqrt, sin, cos, pi, e
import matplotlib.pyplot as plt

def f(p,m):
    a = 0.0000000000529177
    Z = 1
    s = -1 #should be 0 but for conceptual purposes
    A = 1 #A actually represents A'- A
    g_n = 1
    g_e = 1
    return ((Z*(-1)**(s+1)*(A)*g_n*g_e)/ (128*pi**2*a))* p*(e**(p)*(2-p)**(2)-32)/(e**((2 + (m*a/Z))*p))
   
masses = np.logspace(12, 8.8, num = 500) #log of the limits #now, adjusted based on initial graph
for m in masses:
    g = lambda p: f(p, m)
    Particle_Shift = quad(g,0, 350)
    print(Particle_Shift)
  
x = masses
plt.yscale('log')
plt.xscale('log')

Ivals = []
for i in range(len(x)):
    g = lambda p: f(p, x[i]) 
    I = quad(g,0, 350)
    Ivals.append(I[0])
print(x.shape)

Ivals = list(map(abs, Ivals)) #abs Ivals


Ivals_inv = 1 / np.array(Ivals)
Ivals_inv[Ivals == 0] = 0

plt.scatter(x, Ivals_inv) #change Ivals to Ivals_inv for 1/P_S

plt.xlim(xmax=2* 10**12)

plt.ylabel('Particle Shift')
plt.xlabel('Mass')
plt.show()


