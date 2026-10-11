import numpy as np
from matplotlib import pyplot as plt, ticker
import matplotlib
import pandas as pd
import json
import argparse

def load_outcomes(path):
    outcomes = json.load(open(path, "r"))["density"]
    if len(outcomes) > 0 and len(outcomes[0]) == 2:
        outcomes = [["rival", precision, count] for precision, count in outcomes]
    outcomes = pd.DataFrame(outcomes, columns=['tool', 'precision', 'count'])
    return outcomes

def bucket_density(outcomes):
    outcomes = outcomes.copy()
    outcomes['precision'] = np.array(outcomes['precision'], dtype=float)
    outcomes['precision'] = np.floor(outcomes['precision'] / 0.01) * 0.01
    return outcomes.groupby(by=['tool', 'precision'], as_index=False, sort=True).sum()

def lower_precision_percentage(outcomes, tool, threshold=0.2):
    tool_outcomes = outcomes[outcomes["tool"] == tool]
    total = tool_outcomes["count"].sum()
    if total == 0:
        return 0.0
    lower = tool_outcomes.loc[tool_outcomes["precision"] < threshold, "count"].sum()
    return round(lower / total * 100, 2)

def plot_density_cdf(outcomes, args):
    fig, ax = plt.subplots(figsize=(4, 3))

    styles = {
        "optimal": ("orange", "-", "optimal"),
        "rival": ("red", "-.", "rival"),
        "baseline": ("green", "--", "ziv+"),
        "ziv": ("dimgrey", ":", "ziv"),
    }
    for tool in ["optimal", "rival", "baseline", "ziv"]:
        tool_outcomes = outcomes[outcomes["tool"] == tool].copy()
        if tool_outcomes.empty:
            continue
        total = tool_outcomes["count"].sum()
        tool_outcomes["cdf"] = tool_outcomes["count"].cumsum() / total
        x = np.concatenate(([0.0], np.array(tool_outcomes['precision'], dtype=float)))
        y = np.concatenate(([0.0], np.array(tool_outcomes["cdf"], dtype=float)))

        color, linestyle, label = styles[tool]
        ax.step(x, y, where='post', linestyle=linestyle, color=color, linewidth=2, label=label)

    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(-0.05, 1.05)
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.set_ylabel("Fraction of operations")
    ax.set_xlabel("Precision (normalized)")
    ax.yaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.3)
    ax.xaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.3)

    plt.legend(loc="best")
    ax.set_title("Density CDF")
    plt.tight_layout()
    plt.savefig(args.path + "/density_cdf_plot.png", format="png")
    plt.savefig(args.path + "/density_cdf_plot.pdf", format="pdf")
    plt.close(fig)

def plot_density_plots(args):
    outcomes = bucket_density(load_outcomes(args.timeline))

    print("\\newcommand{\\DensityPercentageOfLowerPrecisionReval}{" + str(lower_precision_percentage(outcomes, "rival")) + "}")
    print("\\newcommand{\\DensityPercentageOfLowerPrecisionBaseline}{" + str(lower_precision_percentage(outcomes, "baseline")) + "}")
    print("\\newcommand{\\DensityPercentageOfLowerPrecisionZiv}{" + str(lower_precision_percentage(outcomes, "ziv")) + "}")
    print("\\newcommand{\\DensityPercentageOfLowerPrecisionOptimal}{" + str(lower_precision_percentage(outcomes, "optimal")) + "}")
    print("\\newcommand{\\DensityAdvantageRevalOverBaseline}{" + str(round(lower_precision_percentage(outcomes, "rival") / lower_precision_percentage(outcomes, "baseline"), 2)) + "}")
    print("\\newcommand{\\DensityAdvantageRevalOverZiv}{" + str(round(lower_precision_percentage(outcomes, "rival") / lower_precision_percentage(outcomes, "ziv"), 2)) + "}")

    plot_density_cdf(outcomes, args)

parser = argparse.ArgumentParser(prog='histograms.py', description='Script outputs mixed precision histograms for a Herbie run')
parser.add_argument('-t', '--timeline', dest='timeline', default="report/timeline.json")
parser.add_argument('-o', '--output-path', dest='path', default="report")

args = parser.parse_args()
matplotlib.rcParams.update({'font.size': 12})
plot_density_plots(args)
