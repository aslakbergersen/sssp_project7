import Niederer_et_al_2006 as niederer
from scipy.integrate import odeint
from scipy.optimize import fsolve
import math
import numpy as np
import pylab

# Mechanics model
def f(lambda_):
    e11 = 0.5 * (lambda_**2 - 1)
    e22 = 0.5 * (1/lambda_ - 1)
    T_p = k1 * e11/(a1 - e11)**(b1) * (2 + (b1*e11)/(a1 - e11))
    T_p += -2*k2 * e22/(a2 - e22)**(b2) * (2 + (b2*e22)/(a2 - e22))
    T_p += T_a
    return T_p

# Parameters for the machincs model
a1 = 0.475
a2 = 0.619
b1 = 1.5
b2 = 1.2
k1 = 2.22
k2 = 2.22

# Time variables
T = 300
N = 300
dt = 1.*T/N
step = 10
global_time = np.linspace(0, T, N+1)

# Initial values of mechanics model to the cell model 
lambda_prev = 1
dldt = 0

tenssion_index = niederer.monitor_indices("Tension")
lambda_solution = []

l_list = []
Ta_list = []
t_list = []

for i, t in enumerate(global_time[:-1]):
    # Set initial values
    t_local = np.linspace(t, global_time[i+1], step+1)
    print t
    if i == 0:
        p = (niederer.init_parameter_values(), )
        init = niederer.init_state_values()
    else:
        p = (niederer.init_parameter_values(lambda_=lambda_prev, dExtensionRatiodt=dldt),)
        init = niederer.init_state_values(z=z_prev, Q_1=Q_1_prev, Q_2=Q_2_prev,
                                          Q_3=Q_3_prev, TRPN=TRPN_prev)

    # Solve for 10 timesteps
    s = odeint(niederer.rhs, init, t_local, p)

    # Get laste state
    z_prev, Q_1_prev, Q_2_prev, Q_3_prev, TRPN_prev = s[-1]

    # Get tension
    m = niederer.monitor(s[-1], t_local[-1], p[0])
    T_a = m[tenssion_index]

    # Update solution
    lambda_ = fsolve(f, lambda_prev)
    dldt = (lambda_ - lambda_prev) / dt
    lambda_prev = lambda_

    l_list.append(lambda_)
    Ta_list.append(T_a)
    t_list.append(t_local[-1])

pylab.figure(0)
pylab.plot(l_list,Ta_list)
pylab.figure(1)
pylab.plot(t_list,Ta_list,label='Ta')
pylab.figure(2)
pylab.plot(t_list,l_list,label='lambda')
pylab.show()
