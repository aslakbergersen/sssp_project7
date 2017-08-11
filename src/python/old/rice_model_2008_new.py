# Gotran generated code for the  "rice_model_2008" model
from __future__ import division

def init_state_values(**values):
    """
    Initialize state values
    """
    # Imports
    import numpy as np
    from modelparameters.utils import Range

    # Init values
    # intf=-4.51134525104e-06, TRPNCaL=0.0147730085064,
    # TRPNCaH=0.130660965615, N_NoXB=0.999999959256,
    # P_NoXB=4.07437173989e-08, N=0.99999783454, XBpostr=1.81017564384e-06,
    # XBprer=3.049496488e-07, xXBprer=3.41212828972e-08,
    # xXBpostr=0.00700005394874
    init_values = np.array([-4.51134525104e-06, 0.0147730085064,\
        0.130660965615, 0.999999959256, 4.07437173989e-08, 0.99999783454,\
        1.81017564384e-06, 3.049496488e-07, 3.41212828972e-08,\
        0.00700005394874], dtype=np.float_)

    # State indices and limit checker
    state_ind = dict([("intf",(0, Range())), ("TRPNCaL",(1, Range())),\
        ("TRPNCaH",(2, Range())), ("N_NoXB",(3, Range())), ("P_NoXB",(4,\
        Range())), ("N",(5, Range())), ("XBpostr",(6, Range())),\
        ("XBprer",(7, Range())), ("xXBprer",(8, Range())), ("xXBpostr",(9,\
        Range()))])

    for state_name, value in values.items():
        if state_name not in state_ind:
            raise ValueError("{0} is not a state.".format(state_name))
        ind, range = state_ind[state_name]
        if value not in range:
            raise ValueError("While setting '{0}' {1}".format(state_name,\
                range.format_not_in(value)))

        # Assign value
        init_values[ind] = value

    return init_values

def init_parameter_values(**values):
    """
    Initialize parameter values
    """
    # Imports
    import numpy as np
    from modelparameters.utils import Range

    # Param values
    # Qfapp=6.25, Qgapp=2.5, Qgxb=6.25, Qhb=6.25, Qhf=6.25, fapp=0.5,
    # gapp=0.07, gslmod=6, gxb=0.07, hb=0.4, hbmdc=0, hf=2,
    # hfmdc=5, sigman=1, sigmap=8, xbmodsp=1, KSE=1, PCon_c=0.02,
    # PCon_t=0.002, PExp_c=70, PExp_t=10, SEon=1, SL=1.89999811516,
    # SL_c=2.25, SLmax=2.4, SLmin=1.4, SLrest=1.85, SLset=1.9, dSL=0,
    # fixed_afterload=0, kxb_normalised=120, massf=50, visc=3,
    # Ca_amplitude=1.45, Ca_diastolic=0.09, start_time=5, tau1=20,
    # tau2=110, TmpC=24, len_hbare=0.1, len_thick=1.65, len_thin=1.2,
    # x_0=0.007, Qkn_p=1.6, Qkoff=1.3, Qkon=1.5, Qkp_n=1.6, kn_p=0.5,
    # koffH=0.025, koffL=0.25, koffmod=1, kon=0.05, kp_n=0.05,
    # nperm=15, perm50=0.5, xPsi=2, Trop_conc=70, kxb=120
    init_values = np.array([6.25, 2.5, 6.25, 6.25, 6.25, 0.5, 0.07, 6, 0.07,\
        0.4, 0, 2, 5, 1, 8, 1, 1, 0.02, 0.002, 70, 10, 1, 1.89999811516,\
        2.25, 2.4, 1.4, 1.85, 1.9, 0, 0, 120, 50, 3, 1.45, 0.09, 5, 20, 110,\
        24, 0.1, 1.65, 1.2, 0.007, 1.6, 1.3, 1.5, 1.6, 0.5, 0.025, 0.25, 1,\
        0.05, 0.05, 15, 0.5, 2, 70, 120], dtype=np.float_)

    # Parameter indices and limit checker
    param_ind = dict([("Qfapp", (0, Range())), ("Qgapp", (1, Range())),\
        ("Qgxb", (2, Range())), ("Qhb", (3, Range())), ("Qhf", (4, Range())),\
        ("fapp", (5, Range())), ("gapp", (6, Range())), ("gslmod", (7,\
        Range())), ("gxb", (8, Range())), ("hb", (9, Range())), ("hbmdc",\
        (10, Range())), ("hf", (11, Range())), ("hfmdc", (12, Range())),\
        ("sigman", (13, Range())), ("sigmap", (14, Range())), ("xbmodsp",\
        (15, Range())), ("KSE", (16, Range())), ("PCon_c", (17, Range())),\
        ("PCon_t", (18, Range())), ("PExp_c", (19, Range())), ("PExp_t", (20,\
        Range())), ("SEon", (21, Range())), ("SL", (22, Range())), ("SL_c",\
        (23, Range())), ("SLmax", (24, Range())), ("SLmin", (25, Range())),\
        ("SLrest", (26, Range())), ("SLset", (27, Range())), ("dSL", (28,\
        Range())), ("fixed_afterload", (29, Range())), ("kxb_normalised",\
        (30, Range())), ("massf", (31, Range())), ("visc", (32, Range())),\
        ("Ca_amplitude", (33, Range())), ("Ca_diastolic", (34, Range())),\
        ("start_time", (35, Range())), ("tau1", (36, Range())), ("tau2", (37,\
        Range())), ("TmpC", (38, Range())), ("len_hbare", (39, Range())),\
        ("len_thick", (40, Range())), ("len_thin", (41, Range())), ("x_0",\
        (42, Range())), ("Qkn_p", (43, Range())), ("Qkoff", (44, Range())),\
        ("Qkon", (45, Range())), ("Qkp_n", (46, Range())), ("kn_p", (47,\
        Range())), ("koffH", (48, Range())), ("koffL", (49, Range())),\
        ("koffmod", (50, Range())), ("kon", (51, Range())), ("kp_n", (52,\
        Range())), ("nperm", (53, Range())), ("perm50", (54, Range())),\
        ("xPsi", (55, Range())), ("Trop_conc", (56, Range())), ("kxb", (57,\
        Range()))])

    for param_name, value in values.items():
        if param_name not in param_ind:
            raise ValueError("{0} is not a parameter.".format(param_name))
        ind, range = param_ind[param_name]
        if value not in range:
            raise ValueError("While setting '{0}' {1}".format(param_name,\
                range.format_not_in(value)))

        # Assign value
        init_values[ind] = value

    return init_values

def state_indices(*states):
    """
    State indices
    """
    state_inds = dict([("intf", 0), ("TRPNCaL", 1), ("TRPNCaH", 2),\
        ("N_NoXB", 3), ("P_NoXB", 4), ("N", 5), ("XBpostr", 6), ("XBprer",\
        7), ("xXBprer", 8), ("xXBpostr", 9)])

    indices = []
    for state in states:
        if state not in state_inds:
            raise ValueError("Unknown state: '{0}'".format(state))
        indices.append(state_inds[state])
    if len(indices)>1:
        return indices
    else:
        return indices[0]

def parameter_indices(*params):
    """
    Parameter indices
    """
    param_inds = dict([("Qfapp", 0), ("Qgapp", 1), ("Qgxb", 2), ("Qhb", 3),\
        ("Qhf", 4), ("fapp", 5), ("gapp", 6), ("gslmod", 7), ("gxb", 8),\
        ("hb", 9), ("hbmdc", 10), ("hf", 11), ("hfmdc", 12), ("sigman", 13),\
        ("sigmap", 14), ("xbmodsp", 15), ("KSE", 16), ("PCon_c", 17),\
        ("PCon_t", 18), ("PExp_c", 19), ("PExp_t", 20), ("SEon", 21), ("SL",\
        22), ("SL_c", 23), ("SLmax", 24), ("SLmin", 25), ("SLrest", 26),\
        ("SLset", 27), ("dSL", 28), ("fixed_afterload", 29),\
        ("kxb_normalised", 30), ("massf", 31), ("visc", 32), ("Ca_amplitude",\
        33), ("Ca_diastolic", 34), ("start_time", 35), ("tau1", 36), ("tau2",\
        37), ("TmpC", 38), ("len_hbare", 39), ("len_thick", 40), ("len_thin",\
        41), ("x_0", 42), ("Qkn_p", 43), ("Qkoff", 44), ("Qkon", 45),\
        ("Qkp_n", 46), ("kn_p", 47), ("koffH", 48), ("koffL", 49),\
        ("koffmod", 50), ("kon", 51), ("kp_n", 52), ("nperm", 53), ("perm50",\
        54), ("xPsi", 55), ("Trop_conc", 56), ("kxb", 57)])

    indices = []
    for param in params:
        if param not in param_inds:
            raise ValueError("Unknown param: '{0}'".format(param))
        indices.append(param_inds[param])
    if len(indices)>1:
        return indices
    else:
        return indices[0]

def monitor_indices(*monitored):
    """
    Monitor indices
    """
    monitor_inds = dict([("fappT", 0), ("gapslmd", 1), ("gappT", 2), ("hfmd",\
        3), ("hbmd", 4), ("hfT", 5), ("hbT", 6), ("gxbmd", 7), ("gxbT", 8),\
        ("SSXBprer", 9), ("SSXBpostr", 10), ("Fnordv", 11), ("force", 12),\
        ("active", 13), ("ppforce_t", 14), ("ppforce_c", 15), ("ppforce",\
        16), ("preload", 17), ("afterload", 18), ("beta", 19), ("Cai", 20),\
        ("konT", 21), ("koffLT", 22), ("koffHT", 23), ("dTRPNCaL", 24),\
        ("dTRPNCaH", 25), ("Tropreg", 26), ("permtot", 27), ("inprmt", 28),\
        ("kn_pT", 29), ("kp_nT", 30), ("dXBpostr", 31), ("P", 32),\
        ("dXBprer", 33), ("dutyprer", 34), ("dutypostr", 35), ("dxXBprer",\
        36), ("dxXBpostr", 37), ("FrSBXB", 38), ("dFrSBXB", 39), ("dsovr_ze",\
        40), ("dsovr_cle", 41), ("dlen_sovr", 42), ("dSOVFThin", 43),\
        ("dSOVFThick", 44), ("TropTot", 45), ("dTropTot", 46), ("dforce",\
        47), ("sovr_ze", 48), ("sovr_cle", 49), ("len_sovr", 50),\
        ("SOVFThick", 51), ("SOVFThin", 52), ("dintf_dt", 53),\
        ("dTRPNCaL_dt", 54), ("dTRPNCaH_dt", 55), ("dN_NoXB_dt", 56),\
        ("dP_NoXB_dt", 57), ("dN_dt", 58), ("dXBpostr_dt", 59),\
        ("dXBprer_dt", 60), ("dxXBprer_dt", 61), ("dxXBpostr_dt", 62)])

    indices = []
    for monitor in monitored:
        if monitor not in monitor_inds:
            raise ValueError("Unknown monitored: '{0}'".format(monitor))
        indices.append(monitor_inds[monitor])
    if len(indices)>1:
        return indices
    else:
        return indices[0]

def rhs(states, t, parameters, values=None):
    """
    Compute the right hand side of the rice_model_2008 ODE
    """
    # Imports
    import numpy as np
    import math

    # Assign states
    assert(len(states) == 10)
    TRPNCaL=states[1]; TRPNCaH=states[2]; N_NoXB=states[3]; P_NoXB=states[4];\
        N=states[5]; XBpostr=states[6]; XBprer=states[7]; xXBprer=states[8];\
        xXBpostr=states[9]

    # Assign parameters
    assert(len(parameters) == 58)
    Qfapp=parameters[0]; Qgapp=parameters[1]; Qgxb=parameters[2];\
        Qhb=parameters[3]; Qhf=parameters[4]; fapp=parameters[5];\
        gapp=parameters[6]; gslmod=parameters[7]; gxb=parameters[8];\
        hb=parameters[9]; hbmdc=parameters[10]; hf=parameters[11];\
        hfmdc=parameters[12]; sigman=parameters[13]; sigmap=parameters[14];\
        xbmodsp=parameters[15]; KSE=parameters[16]; PCon_c=parameters[17];\
        PCon_t=parameters[18]; PExp_c=parameters[19]; PExp_t=parameters[20];\
        SEon=parameters[21]; SL=parameters[22]; SL_c=parameters[23];\
        SLrest=parameters[26]; SLset=parameters[27]; dSL=parameters[28];\
        fixed_afterload=parameters[29]; kxb_normalised=parameters[30];\
        Ca_amplitude=parameters[33]; Ca_diastolic=parameters[34];\
        start_time=parameters[35]; tau1=parameters[36]; tau2=parameters[37];\
        TmpC=parameters[38]; len_hbare=parameters[39];\
        len_thick=parameters[40]; len_thin=parameters[41];\
        x_0=parameters[42]; Qkn_p=parameters[43]; Qkoff=parameters[44];\
        Qkon=parameters[45]; Qkp_n=parameters[46]; kn_p=parameters[47];\
        koffH=parameters[48]; koffL=parameters[49]; koffmod=parameters[50];\
        kon=parameters[51]; kp_n=parameters[52]; nperm=parameters[53];\
        perm50=parameters[54]; xPsi=parameters[55]

    # Init return args
    if values is None:
        values = np.zeros((10,), dtype=np.float_)
    else:
        assert isinstance(values, np.ndarray) and values.shape == (10,)

    # Expressions for the Sarcomere geometry component
    sovr_ze = (len_thick/2 if len_thick/2 < SL/2 else SL/2)
    sovr_cle = (len_thin - SL/2 if len_thin - SL/2 > len_hbare/2 else\
        len_hbare/2)
    len_sovr = -sovr_cle + sovr_ze
    SOVFThick = 2*len_sovr/(len_thick - len_hbare)
    SOVFThin = len_sovr/len_thin

    # Expressions for the Thin filament regulation and crossbridge cycling
    # rates component
    fappT = fapp*xbmodsp*math.pow(Qfapp, -37/10 + TmpC/10)
    gapslmd = 1 + gslmod*(1 - SOVFThick)
    gappT = gapp*xbmodsp*math.pow(Qgapp, -37/10 + TmpC/10)*gapslmd
    hfmd = math.exp(-hfmdc*(xXBprer*xXBprer)*math.copysign(1.0,\
        xXBprer)/(x_0*x_0))
    hbmd = math.exp(hbmdc*((-x_0 + xXBpostr)*(-x_0 +\
        xXBpostr))*math.copysign(1.0, -x_0 + xXBpostr)/(x_0*x_0))
    hfT = hf*xbmodsp*math.pow(Qhf, -37/10 + TmpC/10)*hfmd
    hbT = hb*xbmodsp*math.pow(Qhb, -37/10 + TmpC/10)*hbmd
    gxbmd = (math.exp(sigmap*((x_0 - xXBpostr)*(x_0 - xXBpostr))/(x_0*x_0))\
        if xXBpostr < x_0 else math.exp(sigman*((-x_0 + xXBpostr)*(-x_0 +\
        xXBpostr))/(x_0*x_0)))
    gxbT = gxb*xbmodsp*math.pow(Qgxb, -37/10 + TmpC/10)*gxbmd

    # Expressions for the Normalised active and passive force component
    SSXBpostr = fapp*hf/(fapp*gxb + fapp*hb + fapp*hf + gapp*gxb + gapp*hb +\
        gxb*hf)
    Fnordv = kxb_normalised*x_0*SSXBpostr
    force = kxb_normalised*(XBpostr*xXBpostr + XBprer*xXBprer)*SOVFThick
    active = force/Fnordv
    ppforce_t = PCon_t*(-1 + math.exp(PExp_t*math.fabs(SL -\
        SLrest)))*math.copysign(1.0, SL - SLrest)
    ppforce_c = (PCon_c*(-1 + math.exp(PExp_c*math.fabs(SL - SL_c))) if SL >\
        SL_c else 0)
    ppforce = ppforce_c + ppforce_t
    preload = PCon_t*(-1 + math.exp(PExp_t*math.fabs(SLrest -\
        SLset)))*math.copysign(1.0, SLset - SLrest)
    afterload = (KSE*(SLset - SL) if SEon == 1 else fixed_afterload)
    values[0] = -active - ppforce + afterload + preload

    # Expressions for the Equation for simulated calcium transient component
    beta = math.pow(tau1/tau2, -1/(-1 + tau1/tau2)) - math.pow(tau1/tau2,\
        -1/(1 - tau2/tau1))
    Cai = (Ca_diastolic + (Ca_amplitude -\
        Ca_diastolic)*(-math.exp((start_time - t)/tau2) +\
        math.exp((start_time - t)/tau1))/beta if t > start_time else\
        Ca_diastolic)

    # Expressions for the Ca binding to troponin to thin filament regulation
    # component
    konT = kon*math.pow(Qkon, -37/10 + TmpC/10)
    koffLT = koffL*koffmod*math.pow(Qkoff, -37/10 + TmpC/10)
    koffHT = koffH*koffmod*math.pow(Qkoff, -37/10 + TmpC/10)
    dTRPNCaL = -TRPNCaL*koffLT + (1 - TRPNCaL)*Cai*konT
    dTRPNCaH = -TRPNCaH*koffHT + (1 - TRPNCaH)*Cai*konT
    Tropreg = (1 - SOVFThin)*TRPNCaL + SOVFThin*TRPNCaH
    permtot = math.sqrt(math.fabs(1.0/(1 + math.pow(perm50/Tropreg, nperm))))
    inprmt = (1.0/permtot if 1.0/permtot < 100 else 100)
    values[1] = dTRPNCaL
    values[2] = dTRPNCaH
    kn_pT = kn_p*math.pow(Qkn_p, -37/10 + TmpC/10)*permtot
    kp_nT = kp_n*math.pow(Qkp_n, -37/10 + TmpC/10)*inprmt

    # Expressions for the Regulation and crossbridge cycling state equations
    # component
    values[3] = P_NoXB*kp_nT - N_NoXB*kn_pT
    values[4] = N_NoXB*kn_pT - P_NoXB*kp_nT
    dXBpostr = XBprer*hfT - XBpostr*gxbT - XBpostr*hbT
    P = 1 - N - XBpostr - XBprer
    values[5] = P*kp_nT - N*kn_pT
    dXBprer = P*fappT + XBpostr*hbT - XBprer*gappT - XBprer*hfT
    values[6] = dXBpostr
    values[7] = dXBprer

    # Expressions for the Mean strain of strongly bound states component
    dutyprer = (fappT*gxbT + fappT*hbT)/(fappT*gxbT + fappT*hbT + fappT*hfT +\
        gappT*gxbT + gappT*hbT + gxbT*hfT)
    dutypostr = fappT*hfT/(fappT*gxbT + fappT*hbT + fappT*hfT + gappT*gxbT +\
        gappT*hbT + gxbT*hfT)
    dxXBprer = dSL/2 + xPsi*((-x_0 - xXBprer + xXBpostr)*hbT -\
        fappT*xXBprer)/dutyprer
    dxXBpostr = dSL/2 + xPsi*(x_0 - xXBpostr + xXBprer)*hfT/dutypostr
    values[8] = dxXBprer
    values[9] = dxXBpostr

    # Return results
    return values

def monitor(states, t, parameters, monitored=None):
    """
    Computes monitored expressions of the rice_model_2008 ODE
    """
    # Imports
    import numpy as np
    import math

    # Assign states
    assert(len(states) == 10)
    TRPNCaL=states[1]; TRPNCaH=states[2]; N_NoXB=states[3]; P_NoXB=states[4];\
        N=states[5]; XBpostr=states[6]; XBprer=states[7]; xXBprer=states[8];\
        xXBpostr=states[9]

    # Assign parameters
    assert(len(parameters) == 58)
    Qfapp=parameters[0]; Qgapp=parameters[1]; Qgxb=parameters[2];\
        Qhb=parameters[3]; Qhf=parameters[4]; fapp=parameters[5];\
        gapp=parameters[6]; gslmod=parameters[7]; gxb=parameters[8];\
        hb=parameters[9]; hbmdc=parameters[10]; hf=parameters[11];\
        hfmdc=parameters[12]; sigman=parameters[13]; sigmap=parameters[14];\
        xbmodsp=parameters[15]; KSE=parameters[16]; PCon_c=parameters[17];\
        PCon_t=parameters[18]; PExp_c=parameters[19]; PExp_t=parameters[20];\
        SEon=parameters[21]; SL=parameters[22]; SL_c=parameters[23];\
        SLrest=parameters[26]; SLset=parameters[27]; dSL=parameters[28];\
        fixed_afterload=parameters[29]; kxb_normalised=parameters[30];\
        Ca_amplitude=parameters[33]; Ca_diastolic=parameters[34];\
        start_time=parameters[35]; tau1=parameters[36]; tau2=parameters[37];\
        TmpC=parameters[38]; len_hbare=parameters[39];\
        len_thick=parameters[40]; len_thin=parameters[41];\
        x_0=parameters[42]; Qkn_p=parameters[43]; Qkoff=parameters[44];\
        Qkon=parameters[45]; Qkp_n=parameters[46]; kn_p=parameters[47];\
        koffH=parameters[48]; koffL=parameters[49]; koffmod=parameters[50];\
        kon=parameters[51]; kp_n=parameters[52]; nperm=parameters[53];\
        perm50=parameters[54]; xPsi=parameters[55]; Trop_conc=parameters[56];\
        kxb=parameters[57]

    # Init return args
    if monitored is None:
        monitored = np.zeros((63,), dtype=np.float_)
    else:
        assert isinstance(monitored, np.ndarray) and monitored.shape == (63,)

    # Expressions for the Sarcomere geometry component
    monitored[48] = (len_thick/2 if len_thick/2 < SL/2 else SL/2)
    monitored[49] = (len_thin - SL/2 if len_thin - SL/2 > len_hbare/2 else\
        len_hbare/2)
    monitored[50] = -monitored[49] + monitored[48]
    monitored[51] = 2*monitored[50]/(len_thick - len_hbare)
    monitored[52] = monitored[50]/len_thin

    # Expressions for the Thin filament regulation and crossbridge cycling
    # rates component
    monitored[0] = fapp*xbmodsp*math.pow(Qfapp, -37/10 + TmpC/10)
    monitored[1] = 1 + gslmod*(1 - monitored[51])
    monitored[2] = gapp*xbmodsp*math.pow(Qgapp, -37/10 + TmpC/10)*monitored[1]
    monitored[3] = math.exp(-hfmdc*(xXBprer*xXBprer)*math.copysign(1.0,\
        xXBprer)/(x_0*x_0))
    monitored[4] = math.exp(hbmdc*((-x_0 + xXBpostr)*(-x_0 +\
        xXBpostr))*math.copysign(1.0, -x_0 + xXBpostr)/(x_0*x_0))
    monitored[5] = hf*xbmodsp*math.pow(Qhf, -37/10 + TmpC/10)*monitored[3]
    monitored[6] = hb*xbmodsp*math.pow(Qhb, -37/10 + TmpC/10)*monitored[4]
    monitored[7] = (math.exp(sigmap*((x_0 - xXBpostr)*(x_0 -\
        xXBpostr))/(x_0*x_0)) if xXBpostr < x_0 else math.exp(sigman*((-x_0 +\
        xXBpostr)*(-x_0 + xXBpostr))/(x_0*x_0)))
    monitored[8] = gxb*xbmodsp*math.pow(Qgxb, -37/10 + TmpC/10)*monitored[7]

    # Expressions for the Normalised active and passive force component
    monitored[9] = (fapp*gxb + fapp*hb)/(fapp*gxb + fapp*hb + fapp*hf +\
        gapp*gxb + gapp*hb + gxb*hf)
    monitored[10] = fapp*hf/(fapp*gxb + fapp*hb + fapp*hf + gapp*gxb +\
        gapp*hb + gxb*hf)
    monitored[11] = kxb_normalised*x_0*monitored[10]
    monitored[12] = kxb_normalised*(XBpostr*xXBpostr +\
        XBprer*xXBprer)*monitored[51]
    monitored[13] = monitored[12]/monitored[11]
    monitored[14] = PCon_t*(-1 + math.exp(PExp_t*math.fabs(SL -\
        SLrest)))*math.copysign(1.0, SL - SLrest)
    monitored[15] = (PCon_c*(-1 + math.exp(PExp_c*math.fabs(SL - SL_c))) if\
        SL > SL_c else 0)
    monitored[16] = monitored[14] + monitored[15]
    monitored[17] = PCon_t*(-1 + math.exp(PExp_t*math.fabs(SLrest -\
        SLset)))*math.copysign(1.0, SLset - SLrest)
    monitored[18] = (KSE*(SLset - SL) if SEon == 1 else fixed_afterload)
    monitored[53] = -monitored[13] - monitored[16] + monitored[17] +\
        monitored[18]

    # Expressions for the Equation for simulated calcium transient component
    monitored[19] = math.pow(tau1/tau2, -1/(-1 + tau1/tau2)) -\
        math.pow(tau1/tau2, -1/(1 - tau2/tau1))
    monitored[20] = (Ca_diastolic + (Ca_amplitude -\
        Ca_diastolic)*(-math.exp((start_time - t)/tau2) +\
        math.exp((start_time - t)/tau1))/monitored[19] if t > start_time else\
        Ca_diastolic)

    # Expressions for the Ca binding to troponin to thin filament regulation
    # component
    monitored[21] = kon*math.pow(Qkon, -37/10 + TmpC/10)
    monitored[22] = koffL*koffmod*math.pow(Qkoff, -37/10 + TmpC/10)
    monitored[23] = koffH*koffmod*math.pow(Qkoff, -37/10 + TmpC/10)
    monitored[24] = -TRPNCaL*monitored[22] + (1 -\
        TRPNCaL)*monitored[20]*monitored[21]
    monitored[25] = -TRPNCaH*monitored[23] + (1 -\
        TRPNCaH)*monitored[20]*monitored[21]
    monitored[26] = (1 - monitored[52])*TRPNCaL + TRPNCaH*monitored[52]
    monitored[27] = math.sqrt(math.fabs(1.0/(1 +\
        math.pow(perm50/monitored[26], nperm))))
    monitored[28] = (1.0/monitored[27] if 1.0/monitored[27] < 100 else 100)
    monitored[54] = monitored[24]
    monitored[55] = monitored[25]
    monitored[29] = kn_p*math.pow(Qkn_p, -37/10 + TmpC/10)*monitored[27]
    monitored[30] = kp_n*math.pow(Qkp_n, -37/10 + TmpC/10)*monitored[28]

    # Expressions for the Regulation and crossbridge cycling state equations
    # component
    monitored[56] = P_NoXB*monitored[30] - N_NoXB*monitored[29]
    monitored[57] = N_NoXB*monitored[29] - P_NoXB*monitored[30]
    monitored[31] = XBprer*monitored[5] - XBpostr*monitored[6] -\
        XBpostr*monitored[8]
    monitored[32] = 1 - N - XBpostr - XBprer
    monitored[58] = monitored[30]*monitored[32] - N*monitored[29]
    monitored[33] = XBpostr*monitored[6] + monitored[0]*monitored[32] -\
        XBprer*monitored[2] - XBprer*monitored[5]
    monitored[59] = monitored[31]
    monitored[60] = monitored[33]

    # Expressions for the Mean strain of strongly bound states component
    monitored[34] = (monitored[0]*monitored[6] +\
        monitored[0]*monitored[8])/(monitored[0]*monitored[5] +\
        monitored[0]*monitored[6] + monitored[0]*monitored[8] +\
        monitored[2]*monitored[6] + monitored[2]*monitored[8] +\
        monitored[5]*monitored[8])
    monitored[35] = monitored[0]*monitored[5]/(monitored[0]*monitored[5] +\
        monitored[0]*monitored[6] + monitored[0]*monitored[8] +\
        monitored[2]*monitored[6] + monitored[2]*monitored[8] +\
        monitored[5]*monitored[8])
    monitored[36] = dSL/2 + xPsi*((-x_0 - xXBprer + xXBpostr)*monitored[6] -\
        monitored[0]*xXBprer)/monitored[34]
    monitored[37] = dSL/2 + xPsi*(x_0 - xXBpostr +\
        xXBprer)*monitored[5]/monitored[35]
    monitored[61] = monitored[36]
    monitored[62] = monitored[37]

    # Expressions for the Calculation of micromolar per millisecondes of Ca
    # for apparent Ca binding component
    monitored[38] = (XBpostr + XBprer)/(monitored[10] + monitored[9])
    monitored[39] = (monitored[31] + monitored[33])/(monitored[10] +\
        monitored[9])
    monitored[40] = (-0.5*dSL if SL < len_thick else 0)
    monitored[41] = (-0.5*dSL if -SL + 2*len_thin > len_hbare else 0)
    monitored[42] = -monitored[41] + monitored[40]
    monitored[43] = monitored[42]/len_thin
    monitored[44] = 2*monitored[42]/(len_thick - len_hbare)
    monitored[45] = Trop_conc*((1 - monitored[52])*TRPNCaL + ((1 -\
        monitored[38])*TRPNCaL + TRPNCaH*monitored[38])*monitored[52])
    monitored[46] = Trop_conc*((1 - monitored[52])*monitored[24] + ((1 -\
        monitored[38])*TRPNCaL + TRPNCaH*monitored[38])*monitored[43] + ((1 -\
        monitored[38])*monitored[24] + TRPNCaH*monitored[39] +\
        monitored[25]*monitored[38] - TRPNCaL*monitored[39])*monitored[52] -\
        TRPNCaL*monitored[43])
    monitored[47] = kxb*(XBpostr*xXBpostr + XBprer*xXBprer)*monitored[44] +\
        kxb*(XBpostr*monitored[37] + XBprer*monitored[36] +\
        monitored[31]*xXBpostr + monitored[33]*xXBprer)*monitored[51]

    # Return results
    return monitored
