import Niederer_et_al_2006 as niederer
from scipy.integrate import odeint
import math
import numpy as np
import pylab

t = np.linspace(0,100,101)

lm_ = np.linspace(0,10,11)
force_index = niederer.monitor_indices("Ca_i")

init = niederer.init_state_values()
p = (niederer.init_parameter_values(),)
s = odeint(niederer.rhs,init,t,p)
Ca_i = []
for t_ in t:
    m = niederer.monitor(s[-1], t_, p[0])
    Ca_i.append(m[force_index])

pylab.plot(t, Ca_i)
#pylab.semilogx(lm_, Fss)

pylab.show()




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

