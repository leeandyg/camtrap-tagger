"""Command-line interface: camtrap-tagger <folder>."""

import argparse
import os
import re

from speciesnet.geolocation import US_ADMIN1_REGION_TO_ABBREVIATION

from camtrap_tagger import pipeline
from camtrap_tagger.metadata import DeploymentSheetError, load_deployment_sheet
from camtrap_tagger.results import (
    sort_for_review,
    write_csv,
    write_deployments_csv,
    write_json,
    write_species_csv,
)
from camtrap_tagger.visualize import visualize_results

SPECIES_SHOWN = 15
DEPLOYMENTS_SHOWN = 15


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
    parser.add_argument("--recheck", action="store_true",
                        help="rerun images with no detections through the whole-image model; "
                             "reports only confident results (rarely finds a missed animal)")
    parser.add_argument("--deployments", metavar="CSV",
                        help="deployment sheet: deploymentID, folder (relative to the input), and "
                             "optionally latitude, longitude, start, end, locationName")
    parser.add_argument("--file-times", action="store_true",
                        help="use file modified times for images without an EXIF capture time "
                             "(only correct if files were copied off the card with dates kept)")
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

    # Check the sheet before loading any models, so mistakes surface in seconds.
    deployments = None
    if args.deployments:
        try:
            deployments = load_deployment_sheet(args.deployments, source)
        except (OSError, DeploymentSheetError) as e:
            raise SystemExit(f"Error: problems with deployment sheet {args.deployments}:\n"
                             + "\n".join(f"  - {p}" for p in getattr(e, "problems", [str(e)])))

    results, predictions = pipeline.run(
        source,
        backend=args.backend,
        model=args.model,
        country=args.country,
        admin1_region=args.admin1_region,
        threshold=args.threshold,
        recheck=args.recheck,
        crops_dir=os.path.join(out, "crops") if args.backend == "crop" else None,
        deployments=deployments,
        file_time_fallback=args.file_times,
    )

    write_json(results, os.path.join(out, "results.json"))
    write_csv(results, os.path.join(out, "results.csv"), source)
    write_species_csv(results, os.path.join(out, "species_counts.csv"))
    write_deployments_csv(results, os.path.join(out, "deployments.csv"))
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

    species = summary["species"]
    if species:
        print(f"\nSpecies ({len(species)} labels; detections / images):")
        for label, counts in list(species.items())[:SPECIES_SHOWN]:
            print(f"  - {label + ':':28}{counts['detections']:>5} / {counts['images']}")
        if len(species) > SPECIES_SHOWN:
            print(f"  ... {len(species) - SPECIES_SHOWN} more in species_counts.csv")

    deployments = results["deployments"]
    if len(deployments) > 1 or any(d["latitude"] is not None for d in deployments):
        print(f"\nDeployments ({len(deployments)}; images, first – last image, animal detections):")
        for d in deployments[:DEPLOYMENTS_SHOWN]:
            span = f"{(d['first_image'] or '?')[:10]} – {(d['last_image'] or '?')[:10]}"
            print(f"  - {d['deploymentID'] + ':':28}{d['images']:>5}  {span}  {d['animal_detections']}")
        if len(deployments) > DEPLOYMENTS_SHOWN:
            print(f"  ... {len(deployments) - DEPLOYMENTS_SHOWN} more in deployments.csv")

    notes = list(results["warnings"])
    if summary["images_without_time"]:
        notes.append(f"{summary['images_without_time']} image(s) have no EXIF capture time; their "
                     "timestamp is left empty (--file-times uses file modified times instead)")
    if summary["images_with_file_time"]:
        notes.append(f"{summary['images_with_file_time']} image(s) had no EXIF capture time; "
                     "used the file's modified time instead")
    if summary["images_outside_deployment"]:
        notes.append(f"{summary['images_outside_deployment']} image(s) were taken outside their "
                     "deployment's start/end (outside_deployment in results.csv)")
    if notes:
        print("\nWarnings:")
        for note in notes:
            print(f"  - {note}")

    if metadata["geofencing_applied"]:
        print(f"\nGeofencing ({metadata['country']}):")
        print(f"  - Species rolled up:  {summary['geofencing_rollups']}")
    print(f"  - Confidence rollups:  {summary['confidence_rollups']}")

    print(f"\nResults:         {os.path.join(out, 'results.json')} (also results.csv)")
    if visualized:
        print(f"Visualizations:  {os.path.join(out, 'visualizations')} ({visualized} images)")
    print(f"Review folders:  {os.path.join(out, 'review')} "
          f"({review_counts['blank']} blank, {review_counts['unknown']} unknown)")
