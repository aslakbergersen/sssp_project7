import rice_model_2008 as rice
from scipy.integrate import odeint
from scipy.optimize import fsolve
import math
from math import exp
import numpy as np
import pylab

# Mechanics model
def f(lambda_):
    dSLdt = 0.5 * SL0 * (lambda_ - lambda_prev) / dt

    xXBprer = xXBprer_prev + dt*(dSLdt + \
                phi / dutyprer * (-fappT*xXBprer_prev + hbT*(xXBpostr_prev - \
                                                x_0 - xXBprer_prev)))

    xXBpostr = xXBpostr_prev + dt*(dSLdt + \
                phi / dutypostr * (hfT*(xXBprer_prev + x_0 - xXBpostr_prev)))

    # Update tension
    tension = SOVFThick*(XBprer_prev*xXBprer+XBpostr_prev*xXBpostr) / (x_0 * SSXBpostr)

    # Mechanics model
    e11 = 0.5 * (lambda_**2 - 1)
    e22 = 0.5 * (1/lambda_ - 1)
    T_p = k1 * e11/(a1 - e11)**(b1) * (2 + (b1*e11)/(a1 - e11))
    T_p += -2*k2 * e22/(a2 - e22)**(b2) * (2 + (b2*e22)/(a2 - e22))
    T_p += tension*force_scale

    return T_p

# Parameters for the cell model
x_0 = 0.007
phi = 2
SL0 = 1.89999811516
force_scale = 200

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
lambda_ = 1
dldt = SL0 * (lambda_ - lambda_prev) / dt

SOVFThick_index = rice.monitor_indices("SOVFThick")
SSXBpostr_index = rice.monitor_indices("SSXBpostr")
dutyprer_index = rice.monitor_indices("dutyprer")
dutypostr_index = rice.monitor_indices("dutypostr")
hfT_index = rice.monitor_indices("hfT")
hbT_index = rice.monitor_indices("hbT")
fappT_index = rice.monitor_indices("fappT")
active_index = rice.monitor_indices("active")
lambda_solution = []

l_list = []
Ta_list = []
t_list = []

for i, t in enumerate(global_time[:-1]):
    # Set initial values
    t_local = np.linspace(t, global_time[i+1], step+1)
    if i % 100 == 0:
        print "Time", t, "ms"
    if i == 0:
        p = (rice.init_parameter_values(SLmin=2.5),)
        #p = (rice.init_parameter_values(), )
        init = rice.init_state_values()
    else:
        p = (rice.init_parameter_values(SLmin=2.5),)
        init = rice.init_state_values(SL=SL_prev, intf=intf_prev,
                                      TRPNCaH=TRPNCaH_prev,
                                      TRPNCaL=TRPNCaL_prev, N=N_prev, N_NoXB=N_NoXB_prev,
                                      P_NoXB=P_NoXB_prev, XBpostr=XBpostr_prev,
                                      XBprer=XBprer_prev,
                                      xXBpostr=xXBpostr_prev,
                                      xXBprer=xXBprer_prev)

    # Solve for 10 timesteps
    s = odeint(rice.rhs, init, t_local, p)

    # Get last state
    SL_prev, intf_prev, TRPNCaH_prev, TRPNCaL_prev, N_prev, N_NoXB_prev, \
    P_NoXB_prev, XBpostr_prev, XBprer_prev, xXBpostr_prev, xXBprer_prev = s[-1]

    # Get tension
    m = rice.monitor(s[-1], t_local[-1], p[0])
    SOVFThick = m[SOVFThick_index]
    SSXBpostr = m[SSXBpostr_index]
    dutyprer = m[dutyprer_index]
    dutypostr = m[dutypostr_index]
    hfT = m[hfT_index]
    hbT = m[hbT_index]
    fappT = m[fappT_index]
    #lambda_ = SL_prev / SL0
    tension = m[active_index] # Note this is force

    # Update solution
    #lambda_ = fsolve(f, lambda_prev)#+1e-12)
    dldt = SL0 * (lambda_ - lambda_prev) / dt
    #lambda_prev = lambda_

    l_list.append(lambda_*SL0)
    Ta_list.append(tension)
    t_list.append(t_local[-1])

pylab.figure(0)
pylab.plot(l_list,Ta_list)
pylab.xlabel("SL [$\mu m$]")
pylab.ylabel("Scaled normalied active force [-]")
pylab.figure(1)
pylab.plot(t_list,Ta_list)
pylab.ylabel("Scaled normalied active force [-]")
pylab.xlabel("Time [ms]")
pylab.figure(2)
pylab.plot(t_list,l_list)
pylab.xlabel("SL [$\mu m$]")
pylab.xlabel("Time [ms]")
#pylab.ylim([0.8, 1])
pylab.show()
