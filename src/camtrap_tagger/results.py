"""Results format: one flat entry per detection, plus a summary.

Turns SpeciesNet prediction dicts (as produced by `pipeline.run`) into the
camtrap-tagger results file, and sorts images into review folders.
"""

import json
import os
import shutil
from datetime import datetime
from importlib.metadata import version


def parse_taxonomy_label(label):
    """
    Parse a SpeciesNet taxonomy label.

    Returns:
        dict with keys:
            - taxonomy_uuid
            - taxonomy_5part (class;order;family;genus;species) padded to 5 parts
            - taxonomy_raw (original label string)
            - common_name (last non-empty rank)
    """
    result = {
        "taxonomy_uuid": None,
        "taxonomy_5part": None,
        "taxonomy_raw": label,
        "common_name": "unknown"
    }

    if not label:
        return result

    parts = label.split(';')
    if len(parts) >= 6:
        result["taxonomy_uuid"] = parts[0] or None
        tax_parts = parts[1:6]
    else:
        tax_parts = parts[:5]

    while len(tax_parts) < 5:
        tax_parts.append("")

    result["taxonomy_5part"] = ';'.join(tax_parts)

    for part in reversed(tax_parts):
        if part:
            result["common_name"] = part
            break

    return result


def parse_ensemble_prediction(prediction):
    """
    Parse ensemble prediction into a clean common name and status flags.

    Returns:
        dict with keys:
            - clean_name
            - raw
            - is_blank
            - is_unknown
    """
    raw = prediction
    if raw is None:
        return {
            "clean_name": "unknown",
            "raw": None,
            "is_blank": False,
            "is_unknown": False
        }

    raw_str = raw if isinstance(raw, str) else str(raw)
    clean_name = raw_str.split(";")[-1].strip() if ";" in raw_str else raw_str.strip()

    is_blank = raw_str == "blank" or raw_str.endswith(";blank")
    is_unknown = (raw_str == "unknown" or raw_str.endswith(";unknown") or
                  "no cv result" in raw_str)

    if is_unknown:
        clean_name = "unknown"
    elif is_blank:
        clean_name = "blank"

    return {
        "clean_name": clean_name,
        "raw": raw_str,
        "is_blank": is_blank,
        "is_unknown": is_unknown
    }


def derive_rollup_status(clean_prediction, prediction_source, raw_common_name=None):
    """Infer a rollup status string from ensemble output."""
    if clean_prediction == "blank":
        return "blank"
    if clean_prediction == "unknown":
        return "unknown"
    if clean_prediction == "animal":
        return "rolled_to_animal"

    source_lower = (prediction_source or "").lower()
    if "geofence" in source_lower:
        return "geofence_rollup" if "rollup" in source_lower else "geofence_valid"
    if "rollup" in source_lower:
        return "confidence_rollup"
    if source_lower == "detector":
        return "detector"
    if source_lower == "classifier":
        return "valid"

    if raw_common_name and clean_prediction != raw_common_name:
        return "rolled_up"

    return "not_applicable"


def detection_entry(filepath, detection_index, det):
    """Build one results entry from a SpeciesNet detection (or whole-image recheck)."""
    raw = det.get("raw_classifications") or {}
    top = parse_taxonomy_label(raw["classes"][0] if raw.get("classes") else None)
    processed = det.get("processed_classification")
    whole_image = detection_index is None

    entry = {
        "filepath": filepath,
        "detection_index": detection_index,
        "category": det.get("label", "unknown"),
        "prediction": det.get("label", "unknown"),
        "prediction_raw": top["common_name"] if raw else None,
        "ensemble_prediction_raw": None,
        "taxonomy_uuid": top["taxonomy_uuid"],
        "taxonomy_5part": top["taxonomy_5part"],
        "taxonomy_raw": top["taxonomy_raw"],
        "rollup_status": "not_applicable",
        "prediction_source": "detector",
        "species_conf": None,
        "bbox": None if whole_image else det.get("bbox"),
        "detector_conf": None if whole_image else det.get("conf"),
        "crop_filepath": det.get("crop_filepath"),
    }

    # Detections that weren't classified (e.g. humans/vehicles in the crop backend)
    # keep the detector's label.
    if not processed:
        return entry

    parsed = parse_ensemble_prediction(processed.get("class"))
    if parsed["is_blank"]:
        category = "blank"
    elif parsed["is_unknown"]:
        category = "unknown"
    elif parsed["clean_name"] in ("human", "vehicle"):
        category = parsed["clean_name"]
    else:
        category = "animal"

    entry.update({
        "category": category,
        "prediction": parsed["clean_name"],
        "ensemble_prediction_raw": parsed["raw"],
        "rollup_status": derive_rollup_status(
            parsed["clean_name"], processed.get("source"), entry["prediction_raw"]
        ),
        "prediction_source": processed.get("source"),
        "species_conf": processed.get("score"),
    })
    if whole_image and category == "animal":
        entry["note"] = "detector_missed"
    return entry


def image_entries(pred, threshold):
    """All results entries for one SpeciesNet image prediction."""
    filepath = pred["filepath"]

    if "detections" not in pred and pred.get("failures"):
        return [{
            "filepath": filepath,
            "detection_index": None,
            "category": "error",
            "prediction": "error",
            "failures": pred["failures"],
        }]

    entries = [
        detection_entry(filepath, i, det)
        for i, det in enumerate(pred.get("detections") or [])
        if det.get("category") != "0" and det.get("conf", 0) >= threshold
    ]
    if entries:
        return entries

    recheck = pred.get("whole_image_recheck")
    if recheck:
        return [detection_entry(filepath, None, recheck)]

    # Nothing above threshold and no recheck: blank on the detector's word alone.
    return [{
        **detection_entry(filepath, None, {"label": "blank"}),
        "rollup_status": "not_applicable",
        "prediction_source": "detector_only",
    }]


def build_results(predictions, metadata, threshold):
    """Flatten SpeciesNet predictions into the results dict (metadata, predictions, summary)."""
    entries = [e for pred in predictions for e in image_entries(pred, threshold)]

    summary = {
        "total_images": len(predictions),
        "total_detections": len(entries),
        "animals": 0,
        "blanks": 0,
        "humans": 0,
        "vehicles": 0,
        "unknowns": 0,
        "errors": 0,
        "detector_missed": 0,
        "geofencing_rollups": 0,
        "confidence_rollups": 0
    }
    plural = {"animal": "animals", "blank": "blanks", "human": "humans",
              "vehicle": "vehicles", "unknown": "unknowns", "error": "errors"}
    for e in entries:
        summary[plural[e["category"]]] += 1
        if e.get("note") == "detector_missed":
            summary["detector_missed"] += 1
        if e.get("rollup_status") == "confidence_rollup":
            summary["confidence_rollups"] += 1
        if e.get("rollup_status") == "geofence_rollup":
            summary["geofencing_rollups"] += 1

    return {
        "metadata": {
            **metadata,
            "geofencing_applied": bool(metadata.get("country")),
            "speciesnet_version": version("speciesnet"),
            "processed_at": datetime.now().isoformat(),
        },
        "predictions": entries,
        "summary": summary,
    }


def write_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def mirrored_path(filepath, source_root, dest_root):
    """Path under dest_root that mirrors filepath's location under source_root."""
    return os.path.join(dest_root, os.path.relpath(filepath, source_root))


def sort_for_review(results, source_root, review_dir):
    """Copy blank and unknown images into review folders. Returns counts per folder."""
    by_image = {}
    for e in results["predictions"]:
        by_image.setdefault(e["filepath"], []).append(e["category"])

    counts = {"blank": 0, "unknown": 0}
    for filepath, categories in by_image.items():
        if all(c == "blank" for c in categories):
            folder = "blank"
        elif "unknown" in categories:
            folder = "unknown"
        else:
            continue
        dest = mirrored_path(filepath, source_root, os.path.join(review_dir, folder))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        try:
            shutil.copy2(filepath, dest)
            counts[folder] += 1
        except OSError as e:
            print(f"  Warning: Could not copy {filepath}: {e}")
    return counts
