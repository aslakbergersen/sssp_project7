from scipy.integrate import odeint
from scipy.optimize import fsolve
from argparse import ArgumentParser
from math import exp, log
from time import time
import sys
from postprocess import *


def read_command_lines():
    decription = """This is the main program for our project. All of us can
    probolay produce results for our individual tasks from this!"""

    parser = ArgumentParser(description=decription)

    # Required arguments
    require = parser.add_argument_group("required named arguments")

    # Option arguments
    parser.add_argument("-s", "--solid_model", default="holzapfel", type=str,
                                                choices=["holzapfel",
                                                         "usysk",
                                                         "zero-pole",
                                                         "holzapfel_viscous"],
                        help="Type of solid model.")
    parser.add_argument("-c", "--cell_model", default="rice", type=str,
                        choices=["rice"], help="Type of cell model, for now only rice is implemented")
    parser.add_argument("-t", "--dt", default=1, type=float, help="Timestep size in [ms]")
    parser.add_argument("-T", "--time", default=1000, type=float, help="End time in [ms]")
    parser.add_argument("-N", "--number_of_timesteps", default=None, type=int,
                       help="Number of time steps")
    parser.add_argument("--no_verbose", default=None, type=bool, help="Print less")
    parser.add_argument("-r", "--step", default=10, type=int,
                        help="Number of cell solves for one solid solve")
    parser.add_argument("-C", "--coupling", default="FE", type=str,
                        choices=["FE",    # Sundnes et al
                                 "all",   # Reference
                                 "xSL",   # Sundnes et al NOT IMPLEMENTED
                                 "GRL",   # Sundnes et al
                                 "fixed", # Naive, not supposed to run, just for a test
                                 "fixed_projection"], # Effort to use extrapolation as a remedy for the fixed method
                       help="Different types of couplings between the cell model and solid model, cf. Sundnes et al. 2014 for a more thourgh description.")

    args = parser.parse_args()

    if args.no_verbose is not None:
        verbose = False
    else:
        verbose = True

    if args.number_of_timesteps is not None:
        dt = args.time/float(args.number_of_timesteps)
    else:
        dt = args.dt

    return args.solid_model, args.cell_model, args.number_of_timesteps, dt, \
            args.step, args.time, args.coupling, verbose


def main(T, N, dt, step, solid_model, coupling, lambda_prev=1, dldt=0):
    # Other possible standard choises for lambda and dldt
    # lambda_prev = 0.9663
    # dldt = SL0 * (lambda_ - lambda_prev) / dt

    def pasive_tension_usysk(lambda_):
        e11 = 0.5 * (lambda_**2 - 1)
        e22 = 0.5 * (1/lambda_ - 1)
        W = bff*e11**2 + bxx*(e22**2+e22**2)
        T_p = 0.5*K*bff*(lambda_**2-1.)*exp(W)

        return T_p

    def pasive_tension_pole_zero(lambda_):
        # Pole-Zero Mechanics model
        e11 = 0.5 * (lambda_**2 - 1)
        e22 = 0.5 * (1/lambda_ - 1)
        T_p = k1 * e11/(a1 - e11)**(b1) * (2 + (b1*e11)/(a1 - e11))
        T_p += -2*k2 * e22/(a2 - e22)**(b2) * (2 + (b2*e22)/(a2 - e22))

        return T_p

    def pasive_tension_holzapfel(lambda_):
        # Holzapfel mechanics model
        c11 = lambda_**2
        c22 = 1./lambda_
        I1 = c11 + 2.*c22
        I4f = c11

        T_p = (a/2.)*exp(b*(I1-3.)) + af*exp(bf*(I4f-1.)**2)*(I4f-1.)

        return T_p

    def pasive_tension_holzapfel_viscous(lambda_):
        # Holzapfel mechanics model with a viscous term
        c11 = lambda_**2
        c22 = 1./lambda_
        I1 = c11 + 2.*c22
        I4f = c11

        alpha_f = (alpha_f_prev + (dt/eta_f)*mu_f*0.5*log(I4f))/(1. + (dt/eta_f)*mu_f)
        alpha_f_tmp.append(alpha_f)
        T_v = (mu_f/I4f)*(0.5*log(I4f) - alpha_f)

        T_p = (a/2.)*exp(b*(I1-3.)) + af*exp(bf*(I4f-1.)**2)*(I4f-1.)

        return T_p + T_v

    def active_tension_FE(lambda_):
        xXBprer = xXBprer_prev + dt*(0.5*SL0*(lambda_ - lambda_prev)/dt + \
                    phi / dutyprer * (-fappT*xXBprer_prev + hbT*(xXBpostr_prev - \
                    x_0 - xXBprer_prev)))

        xXBpostr = xXBpostr_prev + dt* (0.5*SL0*(lambda_ - lambda_prev)/dt + \
                    phi / dutypostr * (hfT*(xXBprer_prev + x_0 - xXBpostr_prev)))

        tension = SOVFThick*(XBprer_prev*xXBprer+XBpostr_prev*xXBpostr) / (x_0 * SSXBpostr)

        return tension

    def active_tension_all(lambda_):
        p = (rice.init_parameter_values(dSL=SL0*(lambda_- lambda_prev )/dt),)
        init = rice.init_state_values(SL=SL0*lambda_,
                                          intf=intf_prev,
                                          TRPNCaH=TRPNCaH_prev,
                                          TRPNCaL=TRPNCaL_prev, N=N_prev, N_NoXB=N_NoXB_prev,
                                          P_NoXB=P_NoXB_prev, XBpostr=XBpostr_prev,
                                          XBprer=XBprer_prev,
                                          xXBpostr=xXBpostr_prev,
                                          xXBprer=xXBprer_prev)

        ss = odeint(rice.rhs, init, [t, t+dt], p)

        SL_prev1, intf_prev1, TRPNCaH_prev1, TRPNCaL_prev1, N_prev1, N_NoXB_prev1, \
        P_NoXB_prev1, XBpostr_prev1, XBprer_prev1, xXBpostr_prev1, xXBprer_prev1 = ss[-1]

        mm = rice.monitor(s[-1], t+dt, p[0])
        SOVFThick1 = mm[SOVFThick_index]
        SSXBpostr1 = mm[SSXBpostr_index]

        tension = SOVFThick1*(XBprer_prev1*xXBprer_prev1+XBpostr_prev1*xXBpostr_prev1) \
                    / (x_0 * SSXBpostr1)

        return tension

    def active_tension_fixed(lambda_):
        if len(Ta_list) <= 3:
            return active_tension_all(lambda_)
        return tension

    def active_tension_fixed_projection(lambda_):
        if len(Ta_list) <= 3:
            return active_tension_all(lambda_)
        else:
            return 1.5 * Ta_list[-1] - 0.5 * Ta_list[-2]

    def active_tension_xSL(lambda_):
        xSL = 0.5 * SL0 * (lambda_ - 1)
        xXB_prer = dt*phi/dutyprer*fappT*xSL + xSL

    def active_tension_GRL(lambda_):
        a_prer = 0.5 * (lambda_ - lambda_prev)/dt + (phi/dutyprer)*\
                 (-fappT*xXBprer_prev + hbT*(xXBpostr_prev -x_0 -xXBprer_prev) )
        a_postr = 0.5 * (lambda_ - lambda_prev)/dt + (phi/dutypostr)*\
                  ( hfT*(xXBprer_prev + x_0 - xXBpostr_prev) )
        b_prer = -phi*(fappT+hbT)/dutyprer
        b_postr = -phi*hfT/dutypostr

        xXBprer = xXBprer_prev + (a_prer/b_prer)*(exp(b_prer*dt) - 1.0)
        xXBpostr = xXBpostr_prev + (a_postr/b_postr)*(exp(b_postr*dt) - 1.0)

        tension = SOVFThick*(XBprer_prev*xXBprer+XBpostr_prev*xXBpostr) / (x_0 * SSXBpostr)

        return tension

    def f(lambda_):
        number_of_newton_tmp.append(lambda_)
        tension = active_tension(lambda_)
        T_p = pasive_tension(lambda_)
        T_p += tension*force_scale

        return T_p

    # Parameters for Holzapfel mechanics model
    if "holzapfel" in solid_model:
        a = 0.057
        b = 8.094
        af = 21.503
        bf = 15.819
        force_scale = 2000000
        if "viscous" in solid_model:
            alpha_f_prev = 0
            alpha_f_tmp = []
            mu_f = 75.382
            eta_f = 98.157
            pasive_tension = pasive_tension_holzapfel_viscous
        else:
            pasive_tension = pasive_tension_holzapfel

    # Parameters for Usysk mechanics model
    elif solid_model == "usysk":
        bff = 20
        bxx = 4
        K = 0.876
        force_scale = 200
        pasive_tension = pasive_tension_usysk

    # Parameters for zero-pole machincs model
    elif solid_model == "zero-pole":
        a1 = 0.475
        a2 = 0.619
        b1 = 1.5
        b2 = 1.2
        k1 = 2.22
        k2 = 2.22
        force_scale = 200
        pasive_tension = pasive_tension_pole_zero


    if coupling == "FE":
        active_tension = active_tension_FE
    elif coupling == "xSL":
        # TODO: Not implemented
        active_tension = active_tension_xSL
    elif coupling == "GRL":
        active_tension = active_tension_GRL
    elif coupling == "all":
        active_tension = active_tension_all
    elif coupling == "fixed":
        active_tension = active_tension_fixed
    elif coupling == "fixed_projection":
        active_tension = active_tension_fixed_projection

    # Global time steps
    global_time = np.linspace(0, T, N+1)

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
    Ta_list_full = []
    t_list = []
    dldt_list = []
    number_of_newton = []
    number_of_newton_tmp = []
    tension_prev = 0
    tension_prev1 = 0
    tension_prev2 = 0
    tension_prev3 = 0

    start_time = time()
    for i, t in enumerate(global_time[:-1]):
        # Print time
        if i % 100 == 0 and i != 0:
            seconds =  (N-i) * (time()-start_time)/i
            if seconds < 120:
                time_left = "estimated time left: %3.00f seconds" % seconds
            elif 120 <= seconds < 3600:
                minutes = seconds / 60.
                time_left = "estimated time left: %03.02f minutes" % minutes
            else:
                hours = seconds / 3600.
                time_left = "estimated time left: %03.02f hours" % hours

            print "Time", t, "ms", time_left

        # Set initial values
        t_local = np.linspace(t, global_time[i+1], step+1)
        if i == 0:
            p = (rice.init_parameter_values(dSL=dldt),)
            init = rice.init_state_values()
        else:
            p = (rice.init_parameter_values(dSL=dldt),)
            init = rice.init_state_values(SL=SL_prev,
                                          intf=intf_prev,
                                          TRPNCaH=TRPNCaH_prev,
                                          TRPNCaL=TRPNCaL_prev, N=N_prev, N_NoXB=N_NoXB_prev,
                                          P_NoXB=P_NoXB_prev, XBpostr=XBpostr_prev,
                                          XBprer=XBprer_prev,
                                          xXBpostr=xXBpostr_prev,
                                          xXBprer=xXBprer_prev)

        # Solve for "step" number of time steps
        s = odeint(rice.rhs, init, t_local, p) #, mxstep=10, mxordn=4, mxords=4)

        # Get last state
        SL_prev, intf_prev, TRPNCaH_prev, TRPNCaL_prev, N_prev, N_NoXB_prev, \
        P_NoXB_prev, XBpostr_prev, XBprer_prev, xXBpostr_prev, xXBprer_prev = s[-1]

        SL_prev1, intf_prev1, TRPNCaH_prev1, TRPNCaL_prev1, N_prev1, N_NoXB_prev1, \
        P_NoXB_prev1, XBpostr_prev1, XBprer_prev1, xXBpostr_prev1, xXBprer_prev1 = s[-1]

        # Get tension
        m = rice.monitor(s[-1], t_local[-1], p[0])
        SOVFThick = m[SOVFThick_index]
        SSXBpostr = m[SSXBpostr_index]
        dutyprer = m[dutyprer_index]
        dutypostr = m[dutypostr_index]
        hfT = m[hfT_index]
        hbT = m[hbT_index]
        fappT = m[fappT_index]
        tension = m[active_index] # Note this is force

        # Store solution
        Ta_list.append(tension)

        # Update solution
        lambda_ = fsolve(f, lambda_prev)
        dldt = SL0 * (lambda_ - lambda_prev) / dt
        lambda_prev = lambda_
        SL_prev = lambda_*SL0
        if solid_model == "holzapfel_viscous":
            alpha_f_prev = alpha_f_tmp[-1]

        # Store solution
        l_list.append(SL0*lambda_)
        t_list.append(t_local[-1])
        dldt_list.append(dldt)
        number_of_newton.append(len(number_of_newton_tmp))
        number_of_newton_tmp = []

    elapsed = time() - start_time

    return l_list, Ta_list, t_list, dldt_list, number_of_newton, elapsed


if __name__ == "__main__":
    solid_model, cell_model, N, dt, step, T, coupling, verbose = read_command_lines()

    # Parameters for the cell model
    if cell_model == "rice" and coupling != "xSL":
        import rice_model_2008_new_dir as rice
    if cell_model == "rice" and coupling == "xSL":
        raise RuntimeError("Not implemented")
        # TODO: create this file
        import rice_model_2008_xSL as rice

    x_0 = 0.007
    phi = 2
    SL0 = 1.89999811516

    # Time variables
    if N is None:
        N = int(T/dt)

    # Run the program
    l_list, Ta_list, t_list, dldt_list, \
        number_of_newton, elapsed = main(T, N, dt, step, solid_model, coupling)

    if verbose:
        print ""
        print "/"*50
        print "// Run time in seconds %.02f" % elapsed
        print "// Mean number of newton iterations", np.mean(number_of_newton)
        print "/"*50, "\n"

    # Download the reference solution, if it is not there
    rel_path = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(os.path.join(rel_path, "reference", solid_model)):
        try:
            ref_path = "http://folk.uio.no/aslakwb/%s.tar.gz" %s solid_model
            a = os.system("curl %s > %s/reference/%s.tar.gz" % (ref_path, rel_path, solid_model))
            a += os.system("tar -xvf %s/reference/%s.tar.gz" % (solid_model, rel_path, solid_model)
            a += os.system("rm %s/reference/%s.tar.gz" % (rel_path, solid_model)
            assert a != 0, "A system command exited uncorrectly"
        except:
            print("Something went wrong when downloading the refenrece" + \
                  " solution. Please download it manually from" + \
                  " http://folk.uio.no/aslakwb/")
            sys.exit(0)

    l_l2, l_inf, Ta_l2, Ta_inf, dldt_l2, dldt_inf = compute_error(l_list,
                                                                  Ta_list,
                                                                  t_list,
                                                                  dldt_list,
                                                                  dt,
                                                                  solid_model,
                                                                  verbose=verbose)

    # Post prosess
    parameters = dict(cell_model=cell_model,
                      dt=dt,
                      elapsed=elapsed,
                      coupling=coupling,
                      solid_model=solid_model,
                      N=N,
                      T=T,
                      step=step,
                      l_l2=l_l2,
                      l_inf=l_inf,
                      Ta_l2=Ta_l2,
                      Ta_inf=Ta_inf,
                      dldt_l2=dldt_l2,
                      dldt_inf=dldt_inf)

    run_folder = store_results(l_list, Ta_list, t_list, dldt_list, number_of_newton, parameters)
    postprosess(l_list, Ta_list, t_list, dldt_list, number_of_newton, run_folder)
