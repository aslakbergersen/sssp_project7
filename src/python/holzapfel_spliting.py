import rice_model_2008 as rice
from scipy.integrate import odeint
from scipy.optimize import fsolve
import math
from math import exp
import numpy as np
import pylab

# Mechanics model
def f(lambda_):
    # TODO: Update with one solve xXBprer and xXBpost
    tension = XBprer*xXBprer+XBpost*xXBpost  # TODO find the scaling X_max and SOVF
    e11 = 0.5 * (lambda_**2 - 1)
    e22 = 0.5 * (1/lambda_ - 1)
    T_p = k1 * e11/(a1 - e11)**(b1) * (2 + (b1*e11)/(a1 - e11))
    T_p += -2*k2 * e22/(a2 - e22)**(b2) * (2 + (b2*e22)/(a2 - e22))
    T_p += Tension
    return T_p


# Parameters for the machincs model
a1 = 0.475
a2 = 0.619
b1 = 1.5
b2 = 1.2
k1 = 2.22
k2 = 2.22

# Time variables
T = 1000
N = 1000
dt = 1.*T/N
step = 10
global_time = np.linspace(0, T, N+1)

# Initial values of mechanics model to the cell model 
lambda_prev = 1 #0.9663

SL0 = 1.89999811516

pre_index = rice.monitor_indices("xXBprer")
post_index = rice.monitor_indices("xXBpostr")
lambda_solution = []

l_list = []
Ta_list = []
t_list = []

for i, t in enumerate(global_time[:-1]):
    # Set initial values
    t_local = np.linspace(t, global_time[i+1], step+1)
    print t
    if i == 0:
        p = (rice.init_parameter_values(), )
        init = rice.init_state_values()
    else:
        p = (rice.init_parameter_values(lambda_=lambda_prev, dExtensionRatiodt=dldt),)
        init = rice.init_state_values(SL=SL_prev, intf=intf_prev,
                                      TRPMCaH=TRPMCaH_prev,
                                      TRPNCal=TRPNCal_prev, N=N_prev, N_NoXB=N_NoXB_prev,
                                      P_NoZB=P_NoZB_prev, XBpostr=XBpostr_prev,
                                      XBprer=XBprer_prev,
                                      xXBpostr=xXBpostr_prev,
                                      xXBprer=xXBprer_prev)

    # Solve for 10 timesteps
    s = odeint(rice.rhs, init, t_local, p)

    # Get last state
    SL_prev, intf_prev, TRPMCaH_prev, TRPNCal_prev, N_prev, N_NoXB_prev, \
    P_NoZB_prev, XBpostr_prev, XBprer_prev, xXBpostr_prev, xXBprer_prev = s[-1]

    # Get tension
    m = rice.monitor(s[-1], t_local[-1], p[0])
    xXBprer = m[pre_index]
    xXBpost = m[post_index]
    lambda_ = SL_prev / SL0
    #tension = m[tension_index]

    # Update solution
    lambda_ = fsolve(f, lambda_prev+1e-5)
    lambda_prev = lambda_
    dldt = (lambda_ - lambda_prev) / dt

    l_list.append(lambda_)
    Ta_list.append(tension)
    t_list.append(t_local[-1])

pylab.figure(0)
pylab.plot(l_list,Ta_list)
pylab.figure(1)
pylab.plot(t_list,Ta_list,label='Ta')
pylab.figure(2)
pylab.plot(t_list,l_list,label='lambda')
pylab.ylim([0.8, 1])
pylab.show()
