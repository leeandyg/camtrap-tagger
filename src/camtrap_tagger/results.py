"""Results format: one flat entry per detection, plus a summary.

Turns SpeciesNet prediction dicts (as produced by `pipeline.run`) into the
camtrap-tagger results file, and sorts images into review folders.
"""

import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from importlib.metadata import version

from camtrap_tagger.events import DEFAULT_GAP_MINUTES, build_events
from camtrap_tagger.review import DEFAULT_REVIEW_BELOW, review_reasons, review_summary


def parse_taxonomy_label(label):
    """
    Parse a SpeciesNet taxonomy label.

    Returns:
        dict with keys:
            - taxonomy_uuid
            - taxonomy_5part (class;order;family;genus;species) padded to 5 parts
            - taxonomy_raw (original label string)
            - common_name (7th part if present, else last non-empty rank)
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

    if len(parts) >= 7 and parts[6]:
        result["common_name"] = parts[6]
        return result

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
        "detector_label": None if whole_image else det.get("label"),
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


def species_counts(entries, events=()):
    """Animal detections, images and events per label, most detections first.

    Detections count once per photo, so a burst inflates them; events count independent visits.
    """
    detections, images = Counter(), defaultdict(set)
    for e in entries:
        if e["category"] == "animal":
            detections[e["prediction"]] += 1
            images[e["prediction"]].add(e["filepath"])
    in_events = Counter(o["label"] for ev in events for o in ev["observations"]
                        if o["category"] == "animal")
    return {label: {"detections": n, "images": len(images[label]), "events": in_events[label]}
            for label, n in detections.most_common()}


def deployment_summary(deployments, image_info, entries):
    """One row per deployment: its sheet fields plus image and detection counts."""
    rows = []
    for d in deployments:
        paths = [p for p, info in image_info.items() if info["deployment_id"] == d.deployment_id]
        times = sorted(t for p in paths if (t := image_info[p]["timestamp"]))
        rows.append({
            "deploymentID": d.deployment_id,
            "folder": d.folder,
            "latitude": d.latitude,
            "longitude": d.longitude,
            "start": d.start.isoformat() if d.start else None,
            "end": d.end.isoformat() if d.end else None,
            "locationName": d.location_name,
            "images": len(paths),
            "first_image": times[0] if times else None,
            "last_image": times[-1] if times else None,
            "images_outside_deployment": sum(image_info[p]["outside_deployment"] for p in paths),
            "animal_detections": sum(e.get("deployment_id") == d.deployment_id
                                     and e["category"] == "animal" for e in entries),
        })
    return rows


def build_results(predictions, metadata, threshold, image_info=None, deployments=None,
                  warnings=None, event_gap=DEFAULT_GAP_MINUTES, review_below=DEFAULT_REVIEW_BELOW):
    """Flatten SpeciesNet predictions into the results dict.

    image_info maps filepath → deployment_id, timestamp, timestamp_source and
    outside_deployment (from pipeline.run); those fields are added to every entry. Images
    with timestamps are grouped into events `event_gap` minutes apart. Each entry gets
    `review_reasons` (see review.py), using `review_below` as the confidence cutoff.
    """
    image_info = image_info or {}
    entries = [
        {"filepath": e["filepath"], **image_info.get(pred["filepath"], {}), **e}
        for pred in predictions for e in image_entries(pred, threshold)
    ]
    events, event_of = build_events(entries, event_gap)
    for e in entries:
        e["event_id"] = event_of.get(e["filepath"])
        e["review_reasons"] = review_reasons(e, review_below)

    summary = {
        "total_images": len(predictions),
        "total_detections": sum(e["category"] != "error" for e in entries),
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
    summary["species"] = species_counts(entries, events)
    summary["events"] = len(events)
    summary["images_without_time"] = sum(i["timestamp"] is None for i in image_info.values())
    summary["images_with_file_time"] = sum(
        i["timestamp_source"] == "file_modified" for i in image_info.values())
    summary["images_outside_deployment"] = sum(i["outside_deployment"] for i in image_info.values())
    summary["review"] = review_summary(entries)

    return {
        "metadata": {
            **metadata,
            "geofencing_applied": bool(metadata.get("country")),
            "speciesnet_version": version("speciesnet"),
            "processed_at": datetime.now().isoformat(),
        },
        "predictions": entries,
        "summary": summary,
        "deployments": deployment_summary(deployments or [], image_info, entries),
        "events": events,
        "warnings": warnings or [],
    }


def write_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


CSV_COLUMNS = ["file", "deployment_id", "timestamp", "event_id", "detection_index", "category",
               "prediction",
               "species_conf", "detector_conf", "bbox_x", "bbox_y", "bbox_width", "bbox_height",
               "prediction_raw", "rollup_status", "prediction_source", "taxonomy_5part", "note",
               "timestamp_source", "outside_deployment", "review_reasons"]
DEPLOYMENT_COLUMNS = ["deploymentID", "folder", "latitude", "longitude", "start", "end",
                      "locationName", "images", "first_image", "last_image",
                      "images_outside_deployment", "animal_detections"]


def _rounded(value):
    return round(value, 4) if isinstance(value, float) else value


def write_csv(results, path, source_root):
    """Write results.csv: one row per entry, file paths relative to source_root."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for e in results["predictions"]:
            bbox = e.get("bbox") or [None] * 4
            row = {k: _rounded(e.get(k)) for k in CSV_COLUMNS}
            row["file"] = os.path.relpath(e["filepath"], source_root)
            row["review_reasons"] = ";".join(e.get("review_reasons") or [])
            row.update(zip(["bbox_x", "bbox_y", "bbox_width", "bbox_height"], map(_rounded, bbox)))
            writer.writerow(row)


def write_deployments_csv(results, path):
    """Write deployments.csv: one row per deployment with coordinates and counts."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DEPLOYMENT_COLUMNS)
        writer.writeheader()
        writer.writerows(results["deployments"])


def write_events_csv(results, path):
    """Write events.csv: one row per observation (label) per event."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["event_id", "deployment_id", "start", "end", "images", "label",
                         "category", "count", "confidence"])
        for ev in results["events"]:
            for o in ev["observations"]:
                writer.writerow([ev["event_id"], ev["deployment_id"], ev["start"], ev["end"],
                                 ev["images"], o["label"], o["category"], o["count"], o["confidence"]])


def write_species_csv(results, path):
    """Write species_counts.csv: one row per animal label."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["species", "detections", "images", "events"])
        for label, counts in results["summary"]["species"].items():
            writer.writerow([label, counts["detections"], counts["images"], counts["events"]])


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
