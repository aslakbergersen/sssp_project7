import matplotlib.pyplot as plt
from os import listdir, path
import cPickle

coupling = ["GRL", "FE"]
solid_model = ["holzapfel"]
fontsize=20
norm = "inf"
rel_path = path.dirname(path.abspath(__file__))

linewidth=2
color=["b", "k"]

name = {"Ta": "active tension", "l": "sarcomere length", "dldt": "sarcomere velocity"}

for i, var in enumerate(["Ta", "l", "dldt"]):
    plt.figure(i)

    for j, couple in enumerate(coupling):
        runs = listdir(path.join(rel_path, "results", solid_model[0], couple))
        runs.sort(key=lambda x: float(x))

        tmp = []
        dt = []

        for run in runs[1:]:
            sim = listdir(path.join(rel_path, "results", solid_model[0], couple, run))
            sim.sort(key=lambda x: int(x))
            id_ = sim[-1]

            res = open(path.join(rel_path, "results", solid_model[0], couple, run,
                                id_, "data", "params.dat"), "r")
            param = cPickle.load(res)
            if "l_l2" not in param.keys(): continue

            tmp.append(param[var + "_" + norm])
            dt.append(float(run))

        plt.loglog(dt, tmp, "o-", label=couple, linewidth=linewidth, color=color[j])
        plt.hold("on")

    tmp_norm = "$\infty$" if "inf" == norm else "L2"
    plt.title("The %s-error of the %s" % (tmp_norm, name[var]))
    plt.ylabel("Error", fontsize=fontsize)
    plt.xlabel("Time step $\Delta t$ [s]", fontsize=fontsize)
    plt.legend(loc='upper left', #bbox_to_anchor=(0.5, 1.12),
            ncol=1, fancybox=True, shadow=True, fontsize=fontsize)
    plt.savefig("presentation/" + var + "_" + norm + ".eps")
