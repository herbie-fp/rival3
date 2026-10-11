"""Print instruction-repeat statistics from a timing evaluation timeline."""

import argparse
import json


def skipped_percentage(executed, without_skipping):
    if without_skipping == 0:
        return 0.0
    return round((1.0 - executed / without_skipping) * 100, 2)


def print_repeats_statistics(outcomes):
    tools = (
        "rival", "rival-no-repeats", "baseline", "baseline-no-repeats", "ziv"
    )
    counts = {tool: {} for tool in tools}
    for tool, iteration, count in outcomes:
        if iteration > 0 and tool in counts:
            counts[tool][iteration] = count

    rival = counts["rival"]
    rival_no_repeats = counts["rival-no-repeats"]
    baseline = counts["baseline"]
    baseline_no_repeats = counts["baseline-no-repeats"]
    ziv = counts["ziv"]

    average = skipped_percentage(sum(rival.values()), sum(rival_no_repeats.values()))
    print("\\newcommand{\\AveragePercentageOfSkippedInstr}{" + str(average) + "}")

    # As on main, this is the skipped percentage at the last recorded iteration.
    maximum = skipped_percentage(
        rival[max(rival)] if rival else 0,
        rival_no_repeats[max(rival_no_repeats)] if rival_no_repeats else 0,
    )
    print("\\newcommand{\\MaximumPercentageOfSkippedInstr}{" + str(maximum) + "}")

    average_baseline = skipped_percentage(
        sum(baseline.values()), sum(baseline_no_repeats.values())
    )
    print(
        "\\newcommand{\\AveragePercentageOfSkippedInstrBaseline}{"
        + str(average_baseline)
        + "}"
    )

    baseline_total = sum(baseline.values())
    rival_less = (
        round((baseline_total - sum(rival.values())) / baseline_total * 100, 2)
        if baseline_total else 0.0
    )
    print("\\newcommand{\\RivalInstrCountLessThanBaseline}{" + str(rival_less) + "}")

    ziv_total = sum(ziv.values())
    rival_less_than_ziv = (
        round((ziv_total - sum(rival.values())) / ziv_total * 100, 2)
        if ziv_total else 0.0
    )
    print("\\newcommand{\\RivalInstrCountLessThanZiv}{" + str(rival_less_than_ziv) + "}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-t", "--timeline", default="report/timeline.json")
    args = parser.parse_args()
    with open(args.timeline, encoding="utf-8") as timeline_file:
        outcomes = json.load(timeline_file)["instr-executed-cnt"]
    print_repeats_statistics(outcomes)


if __name__ == "__main__":
    main()
