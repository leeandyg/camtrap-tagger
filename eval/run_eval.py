"""Run both backends on a prepared evaluation set and score them, with and without the recheck.

The recheck only adds entries for images with no detections, so "without recheck" is
computed from the same run by dropping those entries; no second run is needed.

    uv run python eval/run_eval.py eval/data/ena24 --country USA
"""

import argparse
import copy
import json
import os
import sys

from speciesnet.utils import ModelInfo

from camtrap_tagger import pipeline
from camtrap_tagger.results import build_results, write_json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import load_taxonomy, score  # noqa: E402


def without_recheck(results, predictions):
    stripped = copy.deepcopy(predictions)
    for p in stripped:
        p.pop("whole_image_recheck", None)
    metadata = {k: v for k, v in results["metadata"].items()
                if k not in ("geofencing_applied", "speciesnet_version", "processed_at")}
    return build_results(stripped, metadata, results["metadata"]["threshold"])


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def markdown_table(rows):
    lines = [
        "| Config | Animals | Exact species | Correct (any level) | Coarser | Animal only | Wrong | "
        "Unknown | Blank | Missed | Recovered by recheck | Extra boxes | Multi-animal images all correct | "
        "Empty images correct | Time (s/img) |",
        "|" + "---|" * 15,
    ]
    for name, m in rows:
        o = m["animal_outcomes"]
        r = m["image_rates"]
        lines.append(
            f"| {name} | {m['truth_animals']} | {pct(m['species_exact_rate'])} | "
            f"{pct(m['correct_any_level_rate'])} | {o['coarser']} | {o['animal']} | {o['wrong']} | "
            f"{o['unknown']} | {o['blank']} | {o['missed']} | {o['recovered']} | "
            f"{sum(m['extra_boxes'].values())} | {pct(r.get('multi'))} | {pct(r.get('empty'))} | "
            f"{m['seconds_per_image']:.2f} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("data", help="folder with ground_truth.json and images/")
    parser.add_argument("--country")
    parser.add_argument("--admin1-region")
    parser.add_argument("--backends", nargs="+", default=["upstream", "crop"])
    parser.add_argument("--out", help="default: <data>/runs")
    parser.add_argument("--rescore", action="store_true",
                        help="score the saved runs in --out again instead of re-running the models")
    args = parser.parse_args()

    with open(os.path.join(args.data, "ground_truth.json")) as f:
        ground_truth = json.load(f)
    taxonomy = load_taxonomy(ModelInfo(pipeline.DEFAULT_MODEL).taxonomy)
    out = args.out or os.path.join(args.data, "runs")

    rows, all_metrics = [], {}
    for backend in args.backends:
        results_path = os.path.join(out, backend, "results.json")
        predictions_path = os.path.join(out, backend, "speciesnet_predictions.json")
        if args.rescore:
            with open(results_path) as f:
                results = json.load(f)
            with open(predictions_path) as f:
                predictions = json.load(f)["predictions"]
        else:
            results, predictions = pipeline.run(
                os.path.join(args.data, "images"), backend=backend, recheck=True,
                country=args.country, admin1_region=args.admin1_region,
            )
            write_json(results, results_path)
            write_json({"predictions": predictions}, predictions_path)

        timing = results["metadata"]["timing_seconds"]
        n = results["summary"]["total_images"]
        for recheck, res in [(True, results), (False, without_recheck(results, predictions))]:
            name = f"{backend} {'+ recheck' if recheck else 'no recheck'}"
            seconds = timing["detect_and_classify"] + (timing.get("recheck", 0) if recheck else 0)
            m = {**score(ground_truth, res, taxonomy), "seconds_per_image": seconds / n,
                 "timing_seconds": timing}
            rows.append((name, m))
            all_metrics[name] = m

    table = markdown_table(rows)
    print("\n" + table)
    write_json({"dataset": ground_truth["dataset"], "country": args.country, "metrics": all_metrics},
               os.path.join(out, "metrics.json"))
    with open(os.path.join(out, "metrics.md"), "w") as f:
        f.write(f"Dataset: {ground_truth['dataset']} ({len(ground_truth['images'])} images)\n\n{table}\n")


if __name__ == "__main__":
    main()
