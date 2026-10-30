"""Export results in the MegaDetector batch output format, for review in Timelapse.

Timelapse (https://timelapse.ucalgary.ca) imports this format as a "recognition file" and
shows each box with the detector's category and our per-animal label, which reviewers can
filter on and correct. Spec: https://lila.science/megadetector-output-format (version 1.6).

Each box keeps the detector's own category (animal / person / vehicle) and confidence; the
final label goes in its "classifications". Paths are relative to the input folder, so open
that folder as the Timelapse image set before importing.
"""

import json
import os
from datetime import datetime

FORMAT_VERSION = "1.6"
DETECTOR_CATEGORIES = {"animal": "1", "human": "2", "vehicle": "3"}


def taxonomy_string(entry):
    """Searchable taxonomy for classification_category_descriptions, e.g. "mammalia;cervidae;..."."""
    ranks = (entry.get("ensemble_prediction_raw") or "").split(";")[1:6]
    return ";".join(r for r in ranks if r) or entry["prediction"]


def build_recognition_file(results, source_root):
    """The MegaDetector-format dict for a results dict."""
    meta = results["metadata"]
    classes, descriptions = {}, {}

    def class_id(entry):
        label = entry["prediction"]
        if label not in classes:
            classes[label] = str(len(classes) + 1)
            descriptions[classes[label]] = taxonomy_string(entry)
        return classes[label]

    by_image = {}
    for e in results["predictions"]:
        by_image.setdefault(e["filepath"], []).append(e)

    images = []
    for path, entries in by_image.items():
        image = {"file": os.path.relpath(path, source_root).replace(os.sep, "/")}
        if entries[0]["category"] == "error":
            image.update(failure="Failure image access", detections=None)
            images.append(image)
            continue

        detections = []
        for e in entries:
            if e["detection_index"] is None and e["category"] == "blank":
                continue  # an empty image, not a box
            if e.get("bbox"):
                detector = e.get("detector_label") or e["category"]
                detection = {"category": DETECTOR_CATEGORIES.get(detector, "1"),
                             "conf": round(e["detector_conf"], 4), "bbox": [round(v, 4) for v in e["bbox"]]}
            else:
                # Whole-image recheck result: no box, so mark the whole frame.
                detection = {"category": "1", "conf": round(e.get("species_conf") or 0.0, 4),
                             "bbox": [0.0, 0.0, 1.0, 1.0]}
            # The final label, including where SpeciesNet overruled the detector (e.g. a
            # "vehicle" box classified as a person). Unclassified boxes have no species_conf.
            if e.get("species_conf") is not None:
                detection["classifications"] = [[class_id(e), round(e["species_conf"], 4)]]
            detections.append(detection)
        image["detections"] = detections
        images.append(image)

    model = meta.get("model", "")
    processed = datetime.fromisoformat(meta["processed_at"]).strftime("%Y-%m-%d %H:%M:%S")
    return {
        "info": {
            "format_version": FORMAT_VERSION,
            "detector": "MDV5A",
            "detection_completion_time": processed,
            "classifier": model,
            "classification_completion_time": processed,
            "detector_metadata": {
                "megadetector_version": "v5a.0.1",
                "typical_detection_threshold": max(meta.get("threshold", 0.1), 0.2),
                "conservative_detection_threshold": meta.get("threshold", 0.1),
            },
        },
        "detection_categories": {v: ("person" if k == "human" else k) for k, v in DETECTOR_CATEGORIES.items()},
        "classification_categories": {v: k for k, v in classes.items()},
        "classification_category_descriptions": descriptions,
        "images": images,
    }


def write_recognition_file(results, path, source_root):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(build_recognition_file(results, source_root), f, indent=1)
