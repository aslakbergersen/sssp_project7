# Gotran generated code for the  "Niederer_et_al_2006" model
from __future__ import division

def init_state_values(**values):
    """
    Initialize state values
    """
    # Imports
    import numpy as np
    from modelparameters.utils import Range

    # Init values
    # z=0.014417937837, Q_1=0, Q_2=0, Q_3=0, TRPN=0.067593139865
    init_values = np.array([0.014417937837, 0, 0, 0, 0.067593139865],\
        dtype=np.float_)

    # State indices and limit checker
    state_ind = dict([("z",(0, Range())), ("Q_1",(1, Range())), ("Q_2",(2,\
        Range())), ("Q_3",(3, Range())), ("TRPN",(4, Range()))])

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
    # beta_0=4.9, Ca_50ref=0.00105, K_z=0.15, alpha_0=0.008, alpha_r1=0.002,
    # alpha_r2=0.00175, beta_1=-4, n_Hill=3, n_Rel=3, z_p=0.85,
    # T_ref=56.2, A_1=-29, A_2=138, A_3=129, a=0.35, alpha_1=0.03,
    # alpha_2=0.13, alpha_3=0.625, Ca_TRPN_Max=0.07, gamma_trpn=2,
    # k_Ref_off=0.2, k_on=100, dExtensionRatiodt=0, lambda_=1
    init_values = np.array([4.9, 0.00105, 0.15, 0.008, 0.002, 0.00175, -4, 3,\
        3, 0.85, 56.2, -29, 138, 129, 0.35, 0.03, 0.13, 0.625, 0.07, 2, 0.2,\
        100, 0, 1], dtype=np.float_)

    # Parameter indices and limit checker
    param_ind = dict([("beta_0", (0, Range())), ("Ca_50ref", (1, Range())),\
        ("K_z", (2, Range())), ("alpha_0", (3, Range())), ("alpha_r1", (4,\
        Range())), ("alpha_r2", (5, Range())), ("beta_1", (6, Range())),\
        ("n_Hill", (7, Range())), ("n_Rel", (8, Range())), ("z_p", (9,\
        Range())), ("T_ref", (10, Range())), ("A_1", (11, Range())), ("A_2",\
        (12, Range())), ("A_3", (13, Range())), ("a", (14, Range())),\
        ("alpha_1", (15, Range())), ("alpha_2", (16, Range())), ("alpha_3",\
        (17, Range())), ("Ca_TRPN_Max", (18, Range())), ("gamma_trpn", (19,\
        Range())), ("k_Ref_off", (20, Range())), ("k_on", (21, Range())),\
        ("dExtensionRatiodt", (22, Range())), ("lambda_", (23, Range()))])

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
    state_inds = dict([("z", 0), ("Q_1", 1), ("Q_2", 2), ("Q_3", 3), ("TRPN",\
        4)])

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
    param_inds = dict([("beta_0", 0), ("Ca_50ref", 1), ("K_z", 2),\
        ("alpha_0", 3), ("alpha_r1", 4), ("alpha_r2", 5), ("beta_1", 6),\
        ("n_Hill", 7), ("n_Rel", 8), ("z_p", 9), ("T_ref", 10), ("A_1", 11),\
        ("A_2", 12), ("A_3", 13), ("a", 14), ("alpha_1", 15), ("alpha_2",\
        16), ("alpha_3", 17), ("Ca_TRPN_Max", 18), ("gamma_trpn", 19),\
        ("k_Ref_off", 20), ("k_on", 21), ("dExtensionRatiodt", 22),\
        ("lambda_", 23)])

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
    monitor_inds = dict([("overlap", 0), ("K_2", 1), ("K_1", 2), ("Ca_50",\
        3), ("Ca_TRPN_50", 4), ("alpha_Tm", 5), ("beta_Tm", 6), ("z_max", 7),\
        ("T_Base", 8), ("Q", 9), ("Tension", 10), ("k_off", 11), ("J_TRPN",\
        12), ("Ca_i", 13), ("Ca_b", 14), ("T_0", 15), ("dz_dt", 16),\
        ("dQ_1_dt", 17), ("dQ_2_dt", 18), ("dQ_3_dt", 19), ("dTRPN_dt", 20)])

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
    Compute the right hand side of the Niederer_et_al_2006 ODE
    """
    # Imports
    import numpy as np
    import math

    # Assign states
    assert(len(states) == 5)
    z, Q_1, Q_2, Q_3, TRPN = states

    # Assign parameters
    assert(len(parameters) == 24)
    beta_0, Ca_50ref, K_z, alpha_0, alpha_r1, alpha_r2, beta_1, n_Hill,\
        n_Rel, z_p, T_ref, A_1, A_2, A_3, a, alpha_1, alpha_2, alpha_3,\
        Ca_TRPN_Max, gamma_trpn, k_Ref_off, k_on, dExtensionRatiodt, lambda_\
        = parameters

    # Init return args
    if values is None:
        values = np.zeros((5,), dtype=np.float_)
    else:
        assert isinstance(values, np.ndarray) and values.shape == (5,)

    # Expressions for the Niederer et al 2006 component
    Ca_i = (0.00018433 if t < 1 else (-0.001356 + 1.055e-06*(t*t*t) +\
        0.0003992*t - 3.507e-05*(t*t) if t < 15 and t >= 10 else (-0.001428 +\
        0.0001494*t + 1.4e-08*(t*t*t) - 2.555e-06*(t*t) if t < 55 and t >= 15 else\
        (0.001719 + 1.739e-11*(t*t*t) - 5.689e-06*t - 3.209e-09*(t*t) if t <\
        250 and t >= 55 else (0.004441 + 1.321e-13*math.pow(t, 4) +\
        1.374e-07*(t*t) - 3.895e-05*t - 2.197e-10*(t*t*t) if t < 490 and t >=\
        250 else 0.00012148)))))
    Ca_b = Ca_TRPN_Max - TRPN

    # Expressions for the Filament overlap component
    overlap = 1 + beta_0*(-1 + lambda_)

    # Expressions for the Tropomyosin component
    K_2 = alpha_r2*math.pow(z_p, n_Rel)*(1 - n_Rel*math.pow(K_z,\
        n_Rel)/(math.pow(K_z, n_Rel) + math.pow(z_p, n_Rel)))/(math.pow(K_z,\
        n_Rel) + math.pow(z_p, n_Rel))
    K_1 = alpha_r2*n_Rel*math.pow(K_z, n_Rel)*math.pow(z_p, -1 +\
        n_Rel)/((math.pow(K_z, n_Rel) + math.pow(z_p, n_Rel))*(math.pow(K_z,\
        n_Rel) + math.pow(z_p, n_Rel)))
    Ca_50 = Ca_50ref*(1 + beta_1*(-1 + lambda_))
    Ca_TRPN_50 = Ca_TRPN_Max*Ca_50/(k_Ref_off*(1 - (0.5 + 0.5*beta_0*(-1 +\
        lambda_))/gamma_trpn)/k_on + Ca_50)
    alpha_Tm = alpha_0*math.pow(Ca_b/Ca_TRPN_50, n_Hill)
    beta_Tm = alpha_r1 + alpha_r2*math.pow(z, -1 + n_Rel)/(math.pow(K_z,\
        n_Rel) + math.pow(z, n_Rel))
    values[0] = (1 - z)*alpha_Tm - beta_Tm*z
    z_max = (-K_2 + alpha_0*math.pow(Ca_TRPN_50/Ca_TRPN_Max,\
        -n_Hill))/(alpha_r1 + alpha_0*math.pow(Ca_TRPN_50/Ca_TRPN_Max,\
        -n_Hill) + K_1)

    # Expressions for the Length independent tension component
    T_Base = T_ref*z/z_max

    # Expressions for the Isometric tension component
    T_0 = T_Base*overlap

    # Expressions for the Cross Bridges component
    Q = Q_1 + Q_2 + Q_3
    Tension = ((1 + a*Q)*T_0/(1 - Q) if Q < 0 else (1 + (2 + a)*Q)*T_0/(1 + Q))
    values[1] = A_1*dExtensionRatiodt - alpha_1*Q_1
    values[2] = A_2*dExtensionRatiodt - alpha_2*Q_2
    values[3] = A_3*dExtensionRatiodt - alpha_3*Q_3

    # Expressions for the Troponin component
    k_off = (k_Ref_off*(1 - Tension/(T_ref*gamma_trpn)) if 1 -\
        Tension/(T_ref*gamma_trpn) > 0.1 else 0.1*k_Ref_off)
    J_TRPN = (Ca_TRPN_Max - TRPN)*k_off - k_on*Ca_i*TRPN

    # Expressions for the Intracellular ion concentrations component
    values[4] = J_TRPN

    # Return results
    return values

def monitor(states, t, parameters, monitored=None):
    """
    Computes monitored expressions of the Niederer_et_al_2006 ODE
    """
    # Imports
    import numpy as np
    import math

    # Assign states
    assert(len(states) == 5)
    z, Q_1, Q_2, Q_3, TRPN = states

    # Assign parameters
    assert(len(parameters) == 24)
    beta_0, Ca_50ref, K_z, alpha_0, alpha_r1, alpha_r2, beta_1, n_Hill,\
        n_Rel, z_p, T_ref, A_1, A_2, A_3, a, alpha_1, alpha_2, alpha_3,\
        Ca_TRPN_Max, gamma_trpn, k_Ref_off, k_on, dExtensionRatiodt, lambda_\
        = parameters

    # Init return args
    if monitored is None:
        monitored = np.zeros((21,), dtype=np.float_)
    else:
        assert isinstance(monitored, np.ndarray) and monitored.shape == (21,)

    # Expressions for the Niederer et al 2006 component
    monitored[13] = (0.00018433 if t < 1 else (-0.001356 + 1.055e-06*(t*t*t)\
        + 0.0003992*t - 3.507e-05*(t*t) if t < 15 and t >= 10 else (-0.001428 +\
        0.0001494*t + 1.4e-08*(t*t*t) - 2.555e-06*(t*t) if t < 55 and t >= 15 else\
        (0.001719 + 1.739e-11*(t*t*t) - 5.689e-06*t - 3.209e-09*(t*t) if t <\
        250 and t >= 55 else (0.004441 + 1.321e-13*math.pow(t, 4) +\
        1.374e-07*(t*t) - 3.895e-05*t - 2.197e-10*(t*t*t) if t < 490 and t >=\
        250 else 0.00012148)))))
    monitored[14] = Ca_TRPN_Max - TRPN

    # Expressions for the Filament overlap component
    monitored[0] = 1 + beta_0*(-1 + lambda_)

    # Expressions for the Tropomyosin component
    monitored[1] = alpha_r2*math.pow(z_p, n_Rel)*(1 - n_Rel*math.pow(K_z,\
        n_Rel)/(math.pow(K_z, n_Rel) + math.pow(z_p, n_Rel)))/(math.pow(K_z,\
        n_Rel) + math.pow(z_p, n_Rel))
    monitored[2] = alpha_r2*n_Rel*math.pow(K_z, n_Rel)*math.pow(z_p, -1 +\
        n_Rel)/((math.pow(K_z, n_Rel) + math.pow(z_p, n_Rel))*(math.pow(K_z,\
        n_Rel) + math.pow(z_p, n_Rel)))
    monitored[3] = Ca_50ref*(1 + beta_1*(-1 + lambda_))
    monitored[4] = Ca_TRPN_Max*monitored[3]/(k_Ref_off*(1 - (0.5 +\
        0.5*beta_0*(-1 + lambda_))/gamma_trpn)/k_on + monitored[3])
    monitored[5] = alpha_0*math.pow(monitored[14]/monitored[4], n_Hill)
    monitored[6] = alpha_r1 + alpha_r2*math.pow(z, -1 + n_Rel)/(math.pow(K_z,\
        n_Rel) + math.pow(z, n_Rel))
    monitored[16] = (1 - z)*monitored[5] - monitored[6]*z
    monitored[7] = (-monitored[1] +\
        alpha_0*math.pow(monitored[4]/Ca_TRPN_Max, -n_Hill))/(alpha_r1 +\
        alpha_0*math.pow(monitored[4]/Ca_TRPN_Max, -n_Hill) + monitored[2])

    # Expressions for the Length independent tension component
    monitored[8] = T_ref*z/monitored[7]

    # Expressions for the Isometric tension component
    monitored[15] = monitored[0]*monitored[8]

    # Expressions for the Cross Bridges component
    monitored[9] = Q_1 + Q_2 + Q_3
    monitored[10] = ((1 + a*monitored[9])*monitored[15]/(1 - monitored[9]) if\
        monitored[9] < 0 else (1 + (2 + a)*monitored[9])*monitored[15]/(1 +\
        monitored[9]))
    monitored[17] = A_1*dExtensionRatiodt - alpha_1*Q_1
    monitored[18] = A_2*dExtensionRatiodt - alpha_2*Q_2
    monitored[19] = A_3*dExtensionRatiodt - alpha_3*Q_3

    # Expressions for the Troponin component
    monitored[11] = (k_Ref_off*(1 - monitored[10]/(T_ref*gamma_trpn)) if 1 -\
        monitored[10]/(T_ref*gamma_trpn) > 0.1 else 0.1*k_Ref_off)
    monitored[12] = (Ca_TRPN_Max - TRPN)*monitored[11] -\
        k_on*TRPN*monitored[13]

    # Expressions for the Intracellular ion concentrations component
    monitored[20] = monitored[12]

    # Return results
    return monitored
