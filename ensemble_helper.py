"""
Helper script to run SpeciesNet ensemble inside the conda environment.

Usage:
    conda run -n speciesnet python ensemble_helper.py --input input.json --output output.json

Input JSON format:
{
    "model_name": "kaggle:google/speciesnet/pyTorch/v4.0.2a/1",
    "batches": [
        {
            "filepaths": ["path/to/image.jpg"],
            "classifier_results": {"path/to/image.jpg": {"classifications": {...}}},
            "detector_results": {"path/to/image.jpg": {"detections": [...]}},
            "geolocation_results": {"path/to/image.jpg": {"country": "USA"}}
        },
        ...
    ]
}

Output JSON format:
{
    "results": [
        {<ensemble output dict for batch 0>},
        {<ensemble output dict for batch 1>},
        ...
    ]
}
"""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Run SpeciesNet ensemble combine")
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    from speciesnet.ensemble import SpeciesNetEnsemble

    model_name = data.get("model_name", "kaggle:google/speciesnet/pyTorch/v4.0.2a/1")
    print(f"  Loading SpeciesNet Ensemble ({model_name})...")
    ensemble = SpeciesNetEnsemble(model_name=model_name)
    print("  - Ensemble loaded successfully")

    results = []
    batches = data.get("batches", [])
    for i, batch in enumerate(batches):
        try:
            output = ensemble.combine(
                filepaths=batch["filepaths"],
                classifier_results=batch["classifier_results"],
                detector_results=batch["detector_results"],
                geolocation_results=batch["geolocation_results"],
                partial_predictions=batch.get("partial_predictions", {}),
            )
            # output is a list; take first element (single-image batch)
            results.append(output[0] if output else {})
        except Exception as e:
            print(f"  Warning: Ensemble failed for batch {i}: {e}")
            results.append({})

    with open(args.output, "w") as f:
        json.dump({"results": results}, f, indent=2)

    print(f"  - Ensemble processed {len(results)} batches")


if __name__ == "__main__":
    main()
