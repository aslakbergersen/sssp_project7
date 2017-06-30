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
    #dSL = (intf_prev + (1 - lambda_prev)*SL0*viscosity) / mass
    #if not SLmin <= SL0*lambda_ and SL0*lambda_ <= SLmax:
    #    dSL = 0

    #xXB_prerss = 0.5*(SL0*(lambda_ - lambda_prev)/dt) + dutyprer/(phi*(

    xXBprer = xXBprer_prev + dt*(#0.5*dSL + \
                                 0.5*SL0*(lambda_ - lambda_prev)/dt + \
                phi / dutyprer * (-fappT*xXBprer_prev + hbT*(xXBpostr_prev - \
                                                x_0 - xXBprer_prev)))


    xXBpostr = xXBpostr_prev + dt* (#0.5*dSL + \
                        0.5*SL0*(lambda_ - lambda_prev)/dt + \
                phi / dutypostr * (hfT*(xXBprer_prev + x_0 - xXBpostr_prev)))

    # Update tension
    tension = SOVFThick*(XBprer_prev*xXBprer+XBpostr_prev*xXBpostr) / (x_0 * SSXBpostr)
    #tension = (XBprer_prev*xXBprer+XBpostr_prev*xXBpostr)
    e11 = 0.5 * (lambda_**2 - 1)
    e22 = 0.5 * (1/lambda_ - 1)
    T_p = k1 * e11/(a1 - e11)**(b1) * (2 + (b1*e11)/(a1 - e11))
    T_p += -2*k2 * e22/(a2 - e22)**(b2) * (2 + (b2*e22)/(a2 - e22))
    #print "Normaliced active force", tension
    #print "Passicve tension", T_p
    #print "SOVFThick", SOVFThick
    #print "SSXBpostr", SSXBpostr
    #print "dutypostr", dutypostr
    #print "dutyprer", dutyprer
    T_p += tension
    #T_p += SOVFThick*(XBprer_prev*xXBprer_prev+XBpostr_prev*xXBpostr_prev) /
    #(x_0 * SSXBpostr)*10  #tension
    return T_p

# Parameters for the cell model
x_0 = 0.007
phi = 2
mass = 0.00025 # For a rabbit
viscosity = 0.003

# Parameters for the machincs model
a1 = 0.475
a2 = 0.619
b1 = 1.5
b2 = 1.2
k1 = 2.22
k2 = 2.22

# Time variables
T = 200
N = 1000
dt = 1.*T/N
step = 10
global_time = np.linspace(0, T, N+1)

# Initial values of mechanics model to the cell model 
lambda_prev = 1 #0.9663

SL0 = 1.89999811516
SLmin = 1.4
SLmax = 2.4

#pre_index = rice.monitor_indices("xXBprer")
#post_index = rice.monitor_indices("xXBpostr")
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
    print t
    if i == 0:
        p = (rice.init_parameter_values(), )
        init = rice.init_state_values()
    else:
        p = (rice.init_parameter_values(),) #SL=lambda_prev*SL0, dExtensionRatiodt=dldt),)
        init = rice.init_state_values(SL=lambda_prev*SL0, intf=intf_prev,
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
    lambda_ = SL_prev / SL0
    tension = m[active_index] # Note this i force

    # Update solution
    lambda_ = fsolve(f, lambda_prev+1e-6)
    lambda_prev = lambda_
    dldt = (lambda_ - lambda_prev) / dt

    l_list.append(lambda_*SL0)
    Ta_list.append(tension)
    t_list.append(t_local[-1])

pylab.figure(0)
pylab.plot(l_list,Ta_list)
pylab.figure(1)
pylab.plot(t_list,Ta_list,label='Ta')
pylab.figure(2)
pylab.plot(t_list,l_list,label='lambda')
#pylab.ylim([0.8, 1])
pylab.show()
