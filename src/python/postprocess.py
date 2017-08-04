import matplotlib.pylab as plt
import os
import cPickle
import numpy as np

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
    plt.xlabel("Time [ms]", fontsize=fontsize)
    plt.savefig(os.path.join(run_folder, "plot", "active_force.eps"))

    plt.figure(2)
    plt.plot(t_list, l_list, linewidth=linewidth, color=color)
    plt.ylabel("SL [$\mu m$]", fontsize=fontsize)
    plt.xlabel("Time [ms]", fontsize=fontsize)
    plt.savefig(os.path.join(run_folder, "plot", "sarcomer_length.eps"))

    plt.figure(3)
    plt.plot(t_list, dldt_list, linewidth=linewidth, color=color)
    plt.ylabel("Shortening velocity [$\mu m/s$]", fontsize=fontsize)
    plt.xlabel("Time [ms]", fontsize=fontsize)
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


def store_results(l_list, Ta_list, t_list, dldt_list, number_of_newton, parameters):
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

    parameters_file = open(os.path.join(run_folder, "data", "params.dat"), "w")
    cPickle.dump(parameters, parameters_file)
    np.array(l_list).dump(os.path.join(run_folder, "data", "length.np"))
    np.array(Ta_list).dump(os.path.join(run_folder, "data", "active_force.np"))
    np.array(t_list).dump(os.path.join(run_folder, "data", "time.np"))
    np.array(dldt_list).dump(os.path.join(run_folder, "data", "velocity.np"))
    np.array(number_of_newton).dump(os.path.join(run_folder, "data", "number_of_newton_iterations.np"))

    return run_folder
