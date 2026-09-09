"""Summarize paired pilot results and human grades; no network calls."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def report(directory: Path, input_price: float | None = None, output_price: float | None = None):
    rows = [json.loads(line) for line in (directory / "results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    with (directory / "grades.csv").open(encoding="utf-8", newline="") as handle:
        grades = {row["id"]: row for row in csv.DictReader(handle)}
    for case_id, grade in grades.items():
        for tier in ("local", "premium"):
            value = (grade.get(f"{tier}_acceptable") or "").strip().lower()
            if value not in {"", "yes", "no", "review"}:
                raise ValueError(f"Invalid {tier} grade for {case_id}: use yes/no/review or blank")
            grade[f"{tier}_acceptable"] = value
    paired = [row for row in rows if all(row["responses"].get(t, {}).get("ok") for t in ("local", "premium"))]
    graded = []
    for row in paired:
        grade = grades.get(row["id"], {})
        if all(grade.get(f"{t}_acceptable", "").strip().lower() in {"yes", "no"} for t in ("local", "premium")):
            graded.append((row, grade))
    print(f"Recorded cases: {len(rows)}; complete successful pairs: {len(paired)}; incomplete/error cases: {len(rows)-len(paired)}")
    print(f"Fully graded successful pairs: {len(graded)} (yes/no for both models only)")
    review_count = sum(any(grades.get(row["id"], {}).get(f"{t}_acceptable") == "review"
                           for t in ("local", "premium")) for row in paired)
    ungraded_count = sum(any(not grades.get(row["id"], {}).get(f"{t}_acceptable")
                             for t in ("local", "premium")) for row in paired)
    print(f"Successful pairs needing review: {review_count}; with missing grades: {ungraded_count} (may overlap)")
    print("Review/blank grades are excluded from quality rates, not counted as passes or failures.")
    if graded:
        counts = {("yes", "yes"): 0, ("no", "yes"): 0, ("yes", "no"): 0, ("no", "no"): 0}
        for _, grade in graded:
            counts[(grade["local_acceptable"], grade["premium_acceptable"])] += 1
        print(f"Resolved comparison pairs: both acceptable={counts[('yes', 'yes')]}; "
              f"premium only={counts[('no', 'yes')]}; local only={counts[('yes', 'no')]}; "
              f"neither acceptable={counts[('no', 'no')]}")
    if not paired:
        return
    print(f"Local routing share on successful pairs: {sum(r['selected_tier']=='local' for r in paired)/len(paired):.1%}")
    for strategy in ("always_local", "always_premium", "router"):
        def choose(row, strategy=strategy):
            return row["selected_tier"] if strategy == "router" else strategy.removeprefix("always_")
        latency = mean(row["responses"][choose(row)]["latency_ms"] +
                       (row["routing_ms"] if strategy == "router" else 0) for row in paired)
        tokens_in = sum(row["responses"]["premium"]["prompt_tokens"] for row in paired if choose(row) == "premium")
        tokens_out = sum(row["responses"]["premium"]["completion_tokens"] for row in paired if choose(row) == "premium")
        quality = (sum(grade[f"{choose(row)}_acceptable"].strip().lower() == "yes" for row, grade in graded)
                   / len(graded)) if graded else None
        text = f"{strategy}: mean sampled latency={latency:.1f}ms; premium tokens in/out={tokens_in}/{tokens_out}"
        text += f"; acceptable={quality:.1%}" if quality is not None else "; acceptable=not graded"
        if input_price is not None and output_price is not None:
            cost = (tokens_in * input_price + tokens_out * output_price) / 1_000_000
            text += f"; estimated premium cost=${cost:.6f}"
        print(text)
    local = [(row, grade) for row, grade in graded if row["selected_tier"] == "local"]
    if local:
        losses = sum(g["local_acceptable"] == "no" and g["premium_acceptable"] == "yes" for _, g in local)
        print(f"Clear local-selection losses vs premium benchmark: {losses}/{len(local)} resolved local-selected pairs")
        print(f"Failure rate among graded local-routed cases: {sum(g['local_acceptable'].strip().lower()=='no' for _,g in local)/len(local):.1%}")
    print("Quality uses resolved yes/no pairs; latency and usage use all successful pairs. Exclusions can bias quality rates.")
    print("Paired-run counterfactual estimates, not load benchmarks or measured gateway end-to-end latency.")
    print("The actual evaluation paid for BOTH providers. Prices exclude local compute, other fees, and missing usage.")
    print("Pilot samples do not establish production reliability. Keep test data separate from rule tuning.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--premium-input-per-million", type=float)
    parser.add_argument("--premium-output-per-million", type=float)
    args = parser.parse_args()
    prices = (args.premium_input_per_million, args.premium_output_per_million)
    if any(p is not None and p < 0 for p in prices) or ((prices[0] is None) != (prices[1] is None)):
        parser.error("Supply both nonnegative prices or neither")
    report(args.directory, *prices)


if __name__ == "__main__":
    main()
