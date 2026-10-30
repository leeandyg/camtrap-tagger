"""Command-line interface: camtrap-tagger <folder>."""

import argparse
import os
import re

from speciesnet.geolocation import US_ADMIN1_REGION_TO_ABBREVIATION

from camtrap_tagger import pipeline
from camtrap_tagger.results import sort_for_review, write_json
from camtrap_tagger.visualize import visualize_results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="camtrap-tagger",
        description="Tag every animal in a folder of camera-trap images with SpeciesNet.",
    )
    parser.add_argument("folder", help="folder of images (searched recursively)")
    parser.add_argument("--out", help="output folder (default: <folder>_tagged next to the input)")
    parser.add_argument("--country", help="ISO 3166-1 alpha-3 country code for geofencing, e.g. USA")
    parser.add_argument("--admin1-region", help="US state code for geofencing, e.g. CA (needs --country USA)")
    parser.add_argument("--backend", choices=["upstream", "crop"], default="upstream",
                        help="how animal detections are classified (default: upstream)")
    parser.add_argument("--threshold", type=float, default=pipeline.DEFAULT_THRESHOLD,
                        help="minimum detector confidence for a detection to count (default: 0.1)")
    parser.add_argument("--model", default=pipeline.DEFAULT_MODEL, help="SpeciesNet crop model")
    parser.add_argument("--no-recheck", action="store_true",
                        help="skip the whole-image recheck of images with no detections")
    parser.add_argument("--no-viz", action="store_true", help="skip drawing boxes on images")
    args = parser.parse_args(argv)

    # SpeciesNet doesn't validate these: a typo like "US" silently disables geofencing.
    if args.country:
        args.country = args.country.upper()
        if not re.fullmatch(r"[A-Z]{3}", args.country):
            parser.error(f"--country must be a 3-letter ISO code like USA, KEN or GBR (got '{args.country}')")
    if args.admin1_region:
        args.admin1_region = args.admin1_region.upper()
        if args.country != "USA":
            parser.error("--admin1-region is only supported with --country USA")
        if args.admin1_region not in US_ADMIN1_REGION_TO_ABBREVIATION.values():
            parser.error(f"--admin1-region must be a US state code like CA or NY (got '{args.admin1_region}')")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    return args


def main(argv=None):
    args = parse_args(argv)
    source = os.path.abspath(args.folder)
    if not os.path.isdir(source):
        raise SystemExit(f"Error: '{args.folder}' is not a folder.")
    out = os.path.abspath(args.out or f"{source}_tagged")

    results, predictions = pipeline.run(
        source,
        backend=args.backend,
        model=args.model,
        country=args.country,
        admin1_region=args.admin1_region,
        threshold=args.threshold,
        recheck=not args.no_recheck,
        crops_dir=os.path.join(out, "crops") if args.backend == "crop" else None,
    )

    results_path = os.path.join(out, "results.json")
    write_json(results, results_path)
    write_json({"predictions": predictions}, os.path.join(out, "speciesnet_predictions.json"))
    review_counts = sort_for_review(results, source, os.path.join(out, "review"))

    visualized = 0
    if not args.no_viz:
        visualized = visualize_results(results, source, os.path.join(out, "visualizations"))

    print_summary(results, out, review_counts, visualized)


def print_summary(results, out, review_counts, visualized):
    summary = results["summary"]
    metadata = results["metadata"]
    print("\n" + "=" * 50)
    print("PROCESSING COMPLETE!")
    print("=" * 50)
    print(f"Total images processed: {summary['total_images']}")
    print(f"Total detections:       {summary['total_detections']}")
    for key in ("animals", "humans", "vehicles", "blanks", "unknowns"):
        print(f"  - {key.capitalize() + ':':20}{summary[key]}")
    if summary["errors"]:
        print(f"  - Errors:             {summary['errors']} (unreadable images)")
    if summary["detector_missed"]:
        print(f"  - Detector missed:    {summary['detector_missed']} (found by whole-image recheck)")

    if metadata["geofencing_applied"]:
        print(f"\nGeofencing ({metadata['country']}):")
        print(f"  - Species rolled up:  {summary['geofencing_rollups']}")
    print(f"  - Confidence rollups:  {summary['confidence_rollups']}")

    print(f"\nResults:         {os.path.join(out, 'results.json')}")
    if visualized:
        print(f"Visualizations:  {os.path.join(out, 'visualizations')} ({visualized} images)")
    print(f"Review folders:  {os.path.join(out, 'review')} "
          f"({review_counts['blank']} blank, {review_counts['unknown']} unknown)")
