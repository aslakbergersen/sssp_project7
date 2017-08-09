import matplotlib.pylab as plt
import os
import cPickle
import numpy as np
from prettytable import PrettyTable

def postprosess(l_list, Ta_list, t_list, dldt_list, number_of_newton, run_folder):
    # Plot parameters
    fontsize = 16
    color = "k"
    linewidth = 2

    # TODO: Add a title for each plot
    plt.figure(0)
    plt.plot(l_list, Ta_list, linewidth=linewidth, color=color)
    plt.xlabel("SL [$\mu m$]", fontsize=fontsize)
    plt.ylabel("Scaled normalied active force [-]", fontsize=fontsize)
    plt.savefig(os.path.join(run_folder, "plot", "sarcomere_length_vs_force.eps"))

    plt.figure(1)
    plt.plot(t_list, Ta_list, linewidth=linewidth, color=color)
    plt.ylabel("Scaled normalied active force [-]", fontsize=fontsize)
    plt.xlabel("Time [s]", fontsize=fontsize)
    plt.savefig(os.path.join(run_folder, "plot", "active_force.eps"))

    plt.figure(2)
    plt.plot(t_list, l_list, linewidth=linewidth, color=color)
    plt.ylabel("SL [$\mu m$]", fontsize=fontsize)
    plt.xlabel("Time [s]", fontsize=fontsize)
    plt.savefig(os.path.join(run_folder, "plot", "sarcomer_length.eps"))

    plt.figure(3)
    plt.plot(t_list, dldt_list, linewidth=linewidth, color=color)
    plt.ylabel("Shortening velocity [$\mu m/s$]", fontsize=fontsize)
    plt.xlabel("Time [s]", fontsize=fontsize)
    plt.savefig(os.path.join(run_folder, "plot", "sarcomere_length_vs_velocity.eps"))

    plt.figure(4)
    plt.plot(Ta_list, dldt_list, linewidth=linewidth, color=color)
    plt.xlabel("Force", fontsize=fontsize)
    plt.ylabel("Shortening velocity [$\mu m/s$]", fontsize=fontsize)
    plt.savefig(os.path.join(run_folder, "plot", "velocity_vs_active_force.eps"))

    plt.figure(5)
    plt.plot(range(len(number_of_newton)), number_of_newton, linewidth=linewidth, color=color)
    plt.xlabel("Time step", fontsize=fontsize)
    plt.ylabel("Number of Newton iterations", fontsize=fontsize)
    plt.savefig(os.path.join(run_folder, "plot", "newton_iteration.eps"))


def store_results(l_list, Ta_list, t_list, dldt_list, number_of_newton,
                  parameters, number_of_substeps, method_type, method_order):
    rel_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    main_folder = os.path.join(rel_path, parameters["solid_model"],
                                parameters["coupling"], str(parameters["dt"]))

    if not os.path.exists(main_folder):
        run_folder = os.path.join(main_folder, "1")
    else:
        runs = os.listdir(main_folder)
        max_run = max([int(f) for f in runs if os.path.isdir(os.path.join(main_folder, f))])
        run_folder = os.path.join(main_folder, str(max_run+1))

    os.makedirs(os.path.join(run_folder, "plot"))
    os.makedirs(os.path.join(run_folder, "data"))
    data_folder = os.path.join(run_folder, "data")

    parameters_file = open(os.path.join(data_folder, "params.dat"), "w")
    cPickle.dump(parameters, parameters_file)
    np.array(l_list).dump(os.path.join(data_folder, "length.np"))
    np.array(Ta_list).dump(os.path.join(data_folder, "active_force.np"))
    np.array(t_list).dump(os.path.join(data_folder, "time.np"))
    np.array(number_of_substeps).dump(os.path.join(data_folder, "number_of_substeps.np"))
    np.array(method_type).dump(os.path.join(data_folder, "method_type.np"))
    np.array(method_order).dump(os.path.join(data_folder, "method_order.np"))
    np.array(dldt_list).dump(os.path.join(data_folder, "velocity.np"))
    np.array(number_of_newton).dump(os.path.join(data_folder, "number_of_newton_iterations.np"))

    return run_folder


def compute_error(l_list, Ta_list, t_list, dldt_list, dt, solid_model, verbose=False):
    # Load reference parameters
    rel_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(rel_path, "reference", solid_model, "data")
    ref_param_file = open(os.path.join(data_path, "params.dat"), "r")
    ref_param = cPickle.load(ref_param_file)

    # Load reference solution
    ref_l = np.load(os.path.join(data_path, "length.np"))
    ref_Ta = np.load(os.path.join(data_path, "active_force.np"))
    ref_t = np.load(os.path.join(data_path, "time.np"))
    ref_dldt = np.load(os.path.join(data_path, "velocity.np"))

    # Convert results to numpy
    l = np.array(l_list)
    Ta = np.array(Ta_list)
    t = np.array(t_list)
    dldt = np.array(dldt_list)

    # Check if we can do a one-to-one comparison
    dt_ratio = dt / ref_param["dt"]
    slice_factor = int(dt_ratio)
    if dt_ratio - slice_factor == 0:
        assert np.sum(np.abs(ref_t[slice_factor-1::slice_factor] - t)) < 1e-9, "The compute error is not correct"
        l_diff = l - ref_l[slice_factor-1::slice_factor]
        Ta_diff = Ta - ref_Ta[slice_factor-1::slice_factor]
        dldt_diff = dldt - ref_dldt[slice_factor-1::slice_factor]

    # Approximate the results with a linear interpolation
    else:
        factor = dt_ratio - slice_factor
        comb_t = (ref_t[slice_factor-1::slice_factor]*(1 - factor) +
                  ref_t[slice_factor-1::slice_factor+1])/2.
        assert np.sum(np.abs(comb_t - t)) < 1e-10, "The compute error is not correct"
        l_diff = l - (ref_l[slice_factor-1::slice_factor]*(1 - factor) +
                      ref_l[slice_factor-1::slice_factor+1]*factor) / 2.
        Ta_diff = Ta - (ref_Ta[slice_factor-1::slice_factor]*(1 - factor) +
                        ref_Ta[slice_factor-1::slice_factor+1]*factor) / 2.
        dldt_diff = dldt - (ref_dtdl[slice_factor-1::slice_factor]*(1 - factor) +
                            ref_dldt[slice_factor-1::slice_factor+1]*factor) / 2.

    l_inf = np.max(np.abs(l_diff))
    l_l2 = np.sqrt(np.sum(l_diff**2)) / l_diff.shape[0]
    Ta_inf = np.max(np.abs(Ta_diff))
    Ta_l2 = np.sqrt(np.sum(Ta_diff**2)) / Ta_diff.shape[0]
    dldt_inf = np.max(np.abs(dldt_diff))
    dldt_l2 = np.sqrt(np.sum(dldt_diff**2)) / dldt_diff.shape[0]

    if verbose:
        table = PrettyTable(["Variabel", "L2-norm", "inf-norm"])

        table.add_row(["Sarcomere length", "%.02e" % l_l2, "%.02e" % l_inf])
        table.add_row(["Active tension", "%.02e" % Ta_l2, "%.02e" % Ta_inf])
        table.add_row(["Velocity", "%.02e" % dldt_l2, "%.02e" % dldt_inf])

        print table

    return l_l2, l_inf, Ta_l2, Ta_inf, dldt_l2, dldt_inf
