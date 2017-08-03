import matplotlib.pylab as plt
import os

def postprosess(l_list, Ta_list, t_list, dldt_list, cell_model, coupling,
                solid_model, dt):
    rel_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot")
    if not os.path.exists(rel_path):
        os.makedirs(rel_path)

    # Plot parameters
    fontsize = 16
    color = "k"
    linewidth = 2

    # TODO: Add a title for each plot
    plt.figure(0)
    plt.plot(l_list, Ta_list, linewidth=linewidth, color=color)
    plt.xlabel("SL [$\mu m$]", fontsize=fontsize)
    plt.ylabel("Scaled normalied active force [-]", fontsize=fontsize)
    plt.savefig(os.path.join(rel_path,
                               "%s_%s_%s_dt%f_sl_force.eps" \
                               % (cell_model, coupling, solid_model, dt)))

    plt.figure(1)
    plt.plot(t_list, Ta_list, linewidth=linewidth, color=color)
    plt.ylabel("Scaled normalied active force [-]", fontsize=fontsize)
    plt.xlabel("Time [ms]", fontsize=fontsize)
    plt.savefig(os.path.join(rel_path,
                               "%s_%s_%s_dt%f_force.eps" \
                               % (cell_model, coupling, solid_model, dt)))

    plt.figure(2)
    plt.plot(t_list, l_list, linewidth=linewidth, color=color)
    plt.ylabel("SL [$\mu m$]", fontsize=fontsize)
    plt.xlabel("Time [ms]", fontsize=fontsize)
    plt.savefig(os.path.join(rel_path,
                               "%s_%s_%s_dt%f_sl.eps" \
                               % (cell_model, coupling, solid_model, dt)))

    plt.figure(3)
    plt.plot(t_list, dldt_list, linewidth=linewidth, color=color)
    plt.ylabel("Shortening velocity [$\mu m/s$]", fontsize=fontsize)
    plt.xlabel("Time [ms]", fontsize=fontsize)
    plt.savefig(os.path.join(rel_path,
                                "%s_%s_%s_dt%f_sl_velocity.eps" \
                                % (cell_model, coupling, solid_model, dt)))

    plt.figure(4)
    plt.plot(Ta_list, dldt_list, linewidth=linewidth, color=color)
    plt.xlabel("Force", fontsize=fontsize)
    plt.ylabel("Shortening velocity [$\mu m/s$]", fontsize=fontsize)
    plt.savefig(os.path.join(rel_path,
                                "%s_%s_%s_dt%f_sl_velocity_force.eps" \
                                % (cell_model, coupling, solid_model, dt)))


