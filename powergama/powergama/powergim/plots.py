import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def read_results_from_excel(filename, sheet):
    """load results from excel into pandas dataframe"""
    df_res = pd.read_excel(filename, sheet_name=sheet)
    return df_res


def plotEnergyMix(
    self,
    model,
    areas=None,
    timeMaxMin=None,
    relative=False,
    showTitle=True,
    variable="energy",
    gentypes=None,
    stage=1,
):
    """
    Plot energy, generation capacity or spilled energy as stacked bars

    Parameters
    ----------
    areas : list of sting
        Which areas to include, default=None means include all
    timeMaxMin : list of two integers
        Time range, [min,max]
    relative : boolean
        Whether to plot absolute (false) or relative (true) values
    variable : string ("energy","capacity","spilled")
        Which variable to plot (default is energy production)
    gentypes : list
        List of generator types to include. None gives all.
    """

    s = stage
    if areas is None:
        areas = list(model.AREA)
    if timeMaxMin is None:
        timeMaxMin = list(model.TIME)
    if gentypes is None:
        gentypes = list(model.GENTYPE)

    gen_output = []
    if variable == "energy":
        print("Getting energy output from all generators...")
        for g in model.GEN:
            gen_output.append(sum(model.generation[g, t, s].value for t in timeMaxMin))
        title = "Energy mix"
    elif variable == "capacity":
        print("Getting capacity from all generators...")
        for g in model.GEN:
            gen_output.append(model.genCapacity[g])
        title = "Capacity mix"
    elif variable == "spilled":
        print("Getting curatailed energy from all generators...")
        for g in model.GEN:
            gen_output.append(sum(self.computeCurtailment(model, g, t, s) for t in timeMaxMin))
        title = "Energy spilled"
    else:
        print("Variable not valid")
        return
    # all_generators = self.grid.getGeneratorsPerAreaAndType()

    if relative:
        prodsum = {}
        for ar in areas:
            prodsum[ar] = 0
            for i in model.GEN:
                if ar == model.nodeArea[model.genNode[i]]:
                    prodsum[ar] += gen_output[i]

    plt.figure()
    ax = plt.subplot(111)
    width = 0.8
    previous = [0] * len(areas)
    numCurves = len(gentypes) + 1
    colours = cm.hsv(np.linspace(0, 1, numCurves))
    # colours = cm.viridis(np.linspace(0, 1, numCurves))
    # colours = cm.Set3(np.linspace(0, 1, numCurves))
    # colours = cm.Grays(np.linspace(0, 1, numCurves))
    # colours = cm.Dark2(np.linspace(0, 1, numCurves))
    count = 0
    ind = range(len(areas))
    for typ in gentypes:
        A = []
        for ar in model.AREA:
            prod = 0
            for g in model.GEN:
                if (typ == model.genType[g]) & (ar == model.nodeArea[model.genNode[g]]):
                    prod += gen_output[g]
                else:
                    prod += 0
            if relative:
                if prodsum[ar] > 0:
                    prod = prod / prodsum[ar]
                    A.append(prod)
                else:
                    A.append(prod)
            else:
                A.append(prod)
        plt.bar(ind, A, width, label=typ, bottom=previous, color=colours[count])
        previous = [previous[i] + A[i] for i in range(len(A))]
        count = count + 1

    handles, labels = ax.get_legend_handles_labels()
    handles.reverse()
    labels.reverse()
    plt.legend(handles, labels, loc="upper right", fontsize="medium")
    #        plt.legend(handles, labels, loc='best',
    #                   bbox_to_anchor=(1.05,1), borderaxespad=0.0)
    plt.xticks(np.arange(len(areas)) + width / 2.0, tuple(areas))
    if showTitle:
        plt.title(title)
    plt.show()
    return


def plotAreaPrice(
    self,
    model,
    boxplot=False,
    areas=None,
    timeMaxMin=None,
    showTitle=False,
    stage=1,
):
    """Show area price(s)
    TODO: incoporate samplefactor

    Parameters
    ----------
    areas (list)
        list of areas to show
    timeMaxMin (list) (default = None)
        [min, max] - lower and upper time interval
    """

    s = stage
    if areas is None:
        areas = list(model.AREA)
    if timeMaxMin is None:
        timeMaxMin = list(model.TIME)
    timerange = range(timeMaxMin[0], timeMaxMin[-1])

    numCurves = len(areas) + 1
    # colours = cm.viridis(np.linspace(0, 1, numCurves))
    colours = cm.hsv(np.linspace(0, 1, numCurves))
    count = 0
    if boxplot:
        areaprice = {}
        factor = {}
        for a in areas:
            areaprice[a] = {}
            factor[a] = {}
            areaprice[a] = [self.computeAreaPrice(model, a, t, s) for t in timerange]
            factor[a] = [model.samplefactor[t] for t in timerange]
        df = pd.DataFrame.from_dict(areaprice)
        props = dict(whiskers="DarkOrange", medians="lime", caps="Gray")
        boxprops = dict(linestyle="--", linewidth=3, color="DarkOrange", facecolor="k")
        flierprops = dict(marker="o", markerfacecolor="none", markersize=8, linestyle="none")
        meanpointprops = dict(marker="D", markeredgecolor="red", markerfacecolor="red")
        medianprops = dict(linestyle="-", linewidth=4, color="red")
        df.plot.box(
            color=props,
            boxprops=boxprops,
            flierprops=flierprops,
            meanprops=meanpointprops,
            medianprops=medianprops,
            patch_artist=True,
            showmeans=True,
        )
        # plt.legend(areas)
    else:
        plt.figure()
        for a in areas:
            areaprice = [self.computeAreaPrice(model, a, t, s) for t in timerange]
            plt.plot(timerange, areaprice, label=a, color=colours[count], lw=2.0)
            count += 1
            if showTitle:
                plt.title("Area price")
    if showTitle:
        plt.title("Area Price")
    plt.legend(loc="upper right", fontsize="medium")
    plt.ylabel("Price [EUR/MWh]")
    plt.show()
    return


def plotWelfare(
    sip_model,
    model,
    areas=None,
    timeMaxMin=None,
    relative=False,
    showTitle=False,
    variable="energy",
    gentypes=None,
    stage=2,
):
    """
    Plot welfare

    Parameters
    ----------
    areas : list of sting
        Which areas to include, default=None means include all
    timeMaxMin : list of two integers
        Time range, [min,max]
    relative : boolean
        Whether to plot absolute (false) or relative (true) values
    variable : string ("energy","capacity","spilled")
        Which variable to plot (default is energy production)
    gentypes : list
        List of generator types to include. None gives all.
    """

    s = stage
    if areas is None:
        areas = []
        for c in model.LOAD:
            areas.append(model.nodeArea[model.demNode[c]])
    if timeMaxMin is None:
        timeMaxMin = list(model.TIME)
    if gentypes is None:
        gentypes = list(model.GENTYPE)

    welfare = {}
    if variable == "all":
        print("Getting welfare from all nodes...")
        types = ["PS", "CS", "CR"]
        for typ in types:
            welfare[typ] = {}
            for c in model.LOAD:
                welfare[typ][c] = (
                    sum([sip_model.computeAreaWelfare(model, c, t, s)[typ] * model.samplefactor[t] for t in model.TIME])
                    / 10 ** 9
                )
        title = "Total welfare"
    else:
        print("Variable not valid")
        return

    if relative:
        total = {}
        for c in model.LOAD:
            total[c] = 0
            for typ in types:
                total[c] += sum(
                    [sip_model.computeAreaWelfare(model, c, t, s)[typ] * model.samplefactor[t] for t in model.TIME]
                )

    plt.figure()
    ax = plt.subplot(111)
    width = 0.8
    previous = [0] * len(areas)
    numCurves = len(types) + 1
    colours = cm.hsv(np.linspace(0, 1, numCurves))
    # colours = cm.viridis(np.linspace(0, 1, numCurves))
    # colours = cm.Set3(np.linspace(0, 1, numCurves))
    # colours = cm.Grays(np.linspace(0, 1, numCurves))
    # colours = cm.Dark2(np.linspace(0, 1, numCurves))
    count = 0
    ind = range(len(model.LOAD))
    for typ in types:
        A = []
        for c in model.LOAD:
            if relative:
                if total[c] > 0:
                    welfare[typ][c] = welfare[typ][c] / total[c]
                    A.append(welfare[typ][c])
            else:
                A.append(welfare[typ][c])
        plt.bar(ind, A, width, label=typ, bottom=previous, color=colours[count])
        previous = [previous[i] + A[i] for i in range(len(A))]
        count = count + 1

    handles, labels = ax.get_legend_handles_labels()
    handles.reverse()
    labels.reverse()
    plt.legend(handles, labels, loc="upper right", fontsize="medium")
    #        plt.legend(handles, labels, loc='best',
    #                   bbox_to_anchor=(1.05,1), borderaxespad=0.0)
    plt.xticks(np.arange(len(areas)) + width / 2.0, tuple(areas))
    plt.ylabel("Annual welfare [bn€]")
    if showTitle:
        plt.title(title)
    plt.show()

    return


def plotInvestments(filename, variable, unit="capacity"):
    """
    Plot investment bar plots

    filename: string
        excel-file generated by 'saveDeterministicResults'
    variable: string
        dcbranch, acbranch, node, generator
    unit: string
        capacity, monetary
    """
    figsize = (8, 6)
    width = 0.8
    if variable == "dcbranch":
        df_res = read_results_from_excel(filename, sheet="branches")
        df_res = df_res[df_res["type"] == "dcdirect"]
        df_res = df_res.groupby(["fArea", "tArea"]).sum()
        numCurves = len(df_res["newCapacity"][df_res["newCapacity"] > 0]) + 1
        colours = cm.hsv(np.linspace(0, 1, numCurves))
        if not df_res["newCapacity"][df_res["newCapacity"] > 0].empty:
            ax1 = df_res["newCapacity"][df_res["newCapacity"] > 0].plot(
                kind="bar",
                title="new capacity",
                figsize=figsize,
                color=colours,
                width=width,
            )
            ax1.set_xlabel("Interconnector", fontsize=12)
            ax1.set_xticklabels(ax1.xaxis.get_majorticklabels(), rotation=0)
            ax1.set_ylabel("New capacity [MW]", fontsize=12)
            ax2 = (
                df_res[["cost_withOM", "congestion_rent"]][df_res["newCapacity"] > 0]
                .divide(10 ** 9)
                .plot(
                    kind="bar",
                    title="costs and benefits",
                    figsize=figsize,
                    legend=True,
                    fontsize=11,
                    color=colours,
                    width=width,
                )
            )
            ax2.set_xlabel("Interconnector", fontsize=12)
            ax2.set_xticklabels(ax2.xaxis.get_majorticklabels(), rotation=0)
            ax2.set_ylabel("Net present value [bn€]", fontsize=12)
    elif variable == "acbranch":
        df_res = read_results_from_excel(filename, sheet="branches")
        df_res = df_res[df_res["type"] == "ac"]
        df_res = df_res.groupby(["fArea", "tArea"]).sum()
        plt.figure()
        df_res["newCapacity"][df_res["newCapacity"] > 0].plot.bar()
    elif variable == "node":
        df_res = read_results_from_excel(filename, sheet="nodes")
    elif variable == "generator":
        df_res = read_results_from_excel(filename, sheet="generation")
        df_res = df_res.groupby(["area"])
        plt.figure()
        df_res["newCapacity"].plot.bar(stacked=True)
    else:
        print("A variable has to be chosen: dcbranch, acbranch, node, generator")

    return


def plotBranchData(sip_model, model, stage=2):
    """
    Plot branch data
    """
    s = stage
    df_branch = pd.DataFrame()
    i = 0
    for b in model.BRANCH:
        for t in model.TIME:
            i += 1
            df_branch.loc[i, "branch"] = b
            df_branch.loc[i, "fArea"] = model.nodeArea[model.branchNodeFrom[b]]
            df_branch.loc[i, "tArea"] = model.nodeArea[model.branchNodeTo[b]]
            df_branch.loc[i, "type"] = model.branchType[b]
            df_branch.loc[i, "hour"] = t
            df_branch.loc[i, "weight"] = model.samplefactor[t]
            df_branch.loc[i, "flow12"] = model.branchFlow12[b, t, s].value
            df_branch.loc[i, "flow21"] = model.branchFlow21[b, t, s].value
            df_branch.loc[i, "utilization"] = (
                model.branchFlow12[b, t, s].value + model.branchFlow21[b, t, s].value
            ) / (
                model.branchExistingCapacity[b]
                + model.branchExistingCapacity2[b]
                + sum(model.branchNewCapacity[b, h + 1].value for h in range(s))
            )

    df_branch.groupby("branch")["flow12"]

    return
