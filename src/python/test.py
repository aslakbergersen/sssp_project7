import Niederer_et_al_2006 as niederer
from scipy.integrate import odeint
from scipy.optimize import fsolve
import math
import numpy as np
import pylab

def f(lambda_):
    a1 = 0.475
    a2 = 0.619
    b1 = 1.5
    b2 = 1.5
    k1 = 2.22
    k2 = 2.22
    

    e11 = 0.5 * (lambda_**2 - 1)
    e22 = 0.5 * (1/lambda_ - 1)
    T_p = k1 * e11/(a1 - e11)**(b1) * (2 + (b1*e11)/(a1 - e11))
    T_p += -2*k2 * e22/(a2 - e22)**(b2) * (2 + (b2*e22)/(a2 - e22))
    T_p += T_a
    return T_p


T = 300
N = 300
dt = 1.*T/N
step = 10

global_time = np.linspace(0, T, N+1)
lambda_prev = 1
dldt = 0
z_prev=0.014417937837
Q_1_prev = 0
Q_2_prev = 0
Q_3_prev = 0
TRPN_prev = 0.067593139865

tenssion_index = niederer.monitor_indices("Tension")
z_index = 0#niederer.monitor_indices("z")
q1_index = 1#niederer.monitor_indices("Tenssion")
q2_index = 2#niederer.monitor_indices("Tenssion")
q3_index =  3#niederer.monitor_indices("Tenssion")
trpn_index = 4 #niederer.monitor_indices("Tenssion")

l_list = []
Ta_list = []
t_list = []

for i, t in enumerate(global_time[0:N]):
    t_local = np.linspace(t, global_time[i+1], step+1)
    print 't = ',t
    p = (niederer.init_parameter_values(lambda_=lambda_prev, dExtensionRatiodt=dldt),)
    init = niederer.init_state_values(z=z_prev, Q_1=Q_1_prev, Q_2=Q_2_prev,
                                      Q_3=Q_3_prev, TRPN=TRPN_prev)

    # Solve
    s = odeint(niederer.rhs, init, t_local, p)

    # Get laste state
    z_prev, Q1_prev, Q2_prev, Q3_prev, TRPN_prev = s[-1]

    # Get tension
    m = niederer.monitor(s[-1], t_local[-1], p[0])
    T_a = m[tenssion_index]

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

#t = np.linspace(0,100,101)
#
#lm_ = np.linspace(0,10,11)
#force_index = niederer.monitor_indices("Ca_i")
#
#init = niederer.init_state_values()
#p = (niederer.init_parameter_values(),)
#s = odeint(niederer.rhs, init, t, p)
#Ca_i = []
#for t_ in t:
#    m = niederer.monitor(s[-1], t_, p[0])
#    Ca_i.append(m[force_index])
#
#pylab.plot(t, Ca_i)
##pylab.semilogx(lm_, Fss)
#
#pylab.show()


#Fmax = 0.17
#Ca50 = 3.0
#n = 7.6
#Fh = Fmax*np.power(Cai,n)/(math.pow(Ca50,n)+np.power(Cai,n))
#pylab.plot(Cai,Fh,'--')

#Fmax =0.92
#Ca50= 0.87
#Fh = Fmax*np.power(Cai,n)/(math.pow(Ca50,n)+np.power(Cai,n))
#pylab.plot(Cai,Fh,'--')



#pylab.show()


#pylab.plot(t,s[:,0],label='force')
#pylab.plot(t,s[:,5],label='N')
#pylab.plot(t,1-s[:,5]-s[:,8]-s[:,9],label='P')
#pylab.plot(t,s[:,8],label='XBpost')
#pylab.plot(t,s[:,9],label='XBpre')
#pylab.legend()
#pylab.show()

