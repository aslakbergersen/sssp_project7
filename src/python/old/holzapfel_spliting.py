import rice_model_2008 as rice
from scipy.integrate import odeint
from scipy.optimize import fsolve
import math
from math import exp
import numpy as np
import pylab

# Mechanics model
def f_usysk(lambda_):    

    xXBprer = xXBprer_prev + dt*(#0.5*dSL + \
                                 0.5*SL0*(lambda_ - lambda_prev)/dt + \
                phi / dutyprer * (-fappT*xXBprer_prev + hbT*(xXBpostr_prev - \
                                                x_0 - xXBprer_prev)))

    xXBpostr = xXBpostr_prev + dt* (#0.5*dSL + \
                        0.5*SL0*(lambda_ - lambda_prev)/dt + \
                phi / dutypostr * (hfT*(xXBprer_prev + x_0 - xXBpostr_prev)))


    # Update tension
    tension = SOVFThick*(XBprer_prev*xXBprer+XBpostr_prev*xXBpostr) / (x_0 * SSXBpostr)

    e11 = 0.5 * (lambda_**2 - 1)
    e22 = 0.5 * (1/lambda_ - 1)
    W = bff*e11**2 + bxx*(e22**2+e22**2)
    T_p = 0.5*K*bff*(lambda_**2-1.)*exp(W)

    T_p += tension*125

    return T_p

# Mechanics model
def f(lambda_):
    # TODO: Update with one solve xXBprer and xXBpost
    #dSL = (intf_prev + (1 - lambda_prev)*SL0*viscosity) / mass
    #if not SLmin <= SL0*lambda_ and SL0*lambda_ <= SLmax:
    #    dSL = 0

    #xXB_prerss = 0.5*(SL0*(lambda_ - lambda_prev)/dt)*dutyprer/(phi*(fappT + hbT)) + (hbT/(fappT+hbT))*(xXBpostr_prev - x_0)
    #tauxXB_prer = dutyprer/(phi*(fappT+hbT))
    #xXB_postrss = 0.5*(SL0*(lambda_ - lambda_prev)/dt)*dutypostr/(phi*hfT) + xXBprer_prev
    #tauxXB_postr = dutypostr/(phi*hfT)

    #xXBprer = xXB_prerss + (xXB_prerss - xXBprer_prev)*exp(-dt/tauxXB_prer)
    #xXBpostr = xXB_postrss + (xXB_postrss - xXBpostr_prev)*exp(-dt/tauxXB_postr)
    

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
    T_p += tension*200.
    #T_p += SOVFThick*(XBprer_prev*xXBprer_prev+XBpostr_prev*xXBpostr_prev) /
    #(x_0 * SSXBpostr)*10  #tension
    return T_p

# Parameters for the cell model
x_0 = 0.007
phi = 2
mass = 0.00025 # For a rabbit
viscosity = 0.003


Qfapp=6.25; Qgapp=2.5; Qgxb=6.25; Qhb=6.25; Qhf=6.25; fapp=0.5
gapp=0.07; gslmod=6; gxb=0.07; hb=0.4; hbmdc=0; hf=2
hfmdc=5; sigman=1; sigmap=8; xbmodsp=1; KSE=1; PCon_c=0.02
PCon_t=0.002; PExp_c=70; PExp_t=10; SEon=1; SL_c=2.25
SLmax=2.4; SLmin=1.4; SLrest=1.85; SLset=1.9; fixed_afterload=0
kxb_normalised=120; massf=50; visc=3; Ca_amplitude=1.45
Ca_diastolic=0.09; start_time=5; tau1=20; tau2=110; TmpC=24
len_hbare=0.1; len_thick=1.65; len_thin=1.2; x_0=0.007
Qkn_p=1.6; Qkoff=1.3; Qkon=1.5; Qkp_n=1.6; kn_p=0.5
koffH=0.025; koffL=0.25; koffmod=1; kon=0.05; kp_n=0.05
nperm=15; perm50=0.5; xPsi=2; Trop_conc=70; kxb=120


# Parameters for the machincs model
a1 = 0.475
a2 = 0.619
b1 = 1.5
b2 = 1.2
k1 = 2.22
k2 = 2.22

bff = 20
bxx = 4
K = 0.876

# Time variables
T = 1000
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
        p = (rice.init_parameter_values(SLmin=2.5), )
        init = rice.init_state_values()
    else:
        p = (rice.init_parameter_values(SLmin=2.5),) #SL=lambda_prev*SL0, dExtensionRatiodt=dldt),)
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
    lambda_ = fsolve(f, lambda_prev)
    lambda_prev = lambda_
    dldt = (lambda_ - lambda_prev) / dt

    l_list.append(SL_prev)
    Ta_list.append(tension)
    t_list.append(t_local[-1])

pylab.figure(0)
pylab.xlabel('lambda')
pylab.ylabel('Ta')
pylab.plot(l_list,Ta_list)
pylab.figure(1)
pylab.xlabel('time')
pylab.ylabel('Ta')
pylab.plot(t_list,Ta_list,label='Ta')
pylab.figure(2)
pylab.xlabel('time')
pylab.ylabel('lambda')
pylab.plot(t_list,l_list,label='lambda')
#pylab.ylim([0.8, 1])
pylab.show()
