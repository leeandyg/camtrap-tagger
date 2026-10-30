"""Flag detections worth a human look, and list their images in review.csv.

SpeciesNet already hedges: it only names a species when its score clears 0.65–0.8, and
otherwise falls back to a broader label ("cervidae family", "animal") or "unknown". So a
confidence cutoff alone misses most uncertain detections; broader labels are flagged too.
"""

import csv
import os

DEFAULT_REVIEW_BELOW = 0.8

# Most to least urgent; the order reasons are listed in.
REASONS = {
    "unknown": "no label at all",
    "detector_disagrees": "classifier overruled the detector (e.g. an animal box labelled human)",
    "frames_disagree": "frames of this video gave different animal labels",
    "recheck_only": "found only by the whole-image recheck, with no detector box",
    "outside_range": "top species isn't expected at this location, so the label was broadened",
    "not_species_level": "labelled above species (genus, family, ... or just 'animal')",
    "low_confidence": "label confidence below the review threshold",
}


def is_species_level(entry):
    """Whether an animal entry's final label names a species ("uuid;class;...;species;common")."""
    parts = (entry.get("ensemble_prediction_raw") or "").split(";")
    return len(parts) == 7 and bool(parts[5])


def review_reasons(entry, below=DEFAULT_REVIEW_BELOW):
    """Why one results entry needs review, most urgent first; [] if it doesn't.

    Blanks aren't flagged: empty images are already copied to review/blank/.
    """
    category = entry["category"]
    if category in ("blank", "error"):
        return []
    reasons = []
    if category == "unknown":
        reasons.append("unknown")
    detector = entry.get("detector_label")
    if detector and category in ("animal", "human", "vehicle") and category != detector:
        reasons.append("detector_disagrees")
    if entry.get("frames_disagree"):
        reasons.append("frames_disagree")
    if entry.get("note") == "detector_missed":
        reasons.append("recheck_only")
    if category == "animal" and not is_species_level(entry):
        geofenced = entry.get("rollup_status") == "geofence_rollup"
        reasons.append("outside_range" if geofenced else "not_species_level")
    conf = entry.get("species_conf")
    if category != "unknown" and conf is not None and conf < below:
        reasons.append("low_confidence")
    return reasons


def review_summary(entries):
    """Counts for the results summary: flagged images and detections, and detections per reason."""
    flagged = [e for e in entries if e.get("review_reasons")]
    by_reason = {r: sum(r in e["review_reasons"] for e in flagged) for r in REASONS}
    return {"images": len({e["filepath"] for e in flagged}), "detections": len(flagged),
            "by_reason": {r: n for r, n in by_reason.items() if n}}


REVIEW_COLUMNS = ["file", "deployment_id", "timestamp", "event_id", "reasons", "lowest_conf",
                  "detections"]


def review_rows(results, source_root):
    """One row per image with a flagged detection, in deployment and capture-time order.

    `detections` lists every non-blank detection in the image, flagged or not, so the row
    stands on its own. Broadened labels show the model's top guess:
    "0: cat family 1.00 (top guess: tiger) [outside_range]; 1: human 0.99".
    """
    by_image = {}
    for e in results["predictions"]:
        by_image.setdefault(e["filepath"], []).append(e)

    rows = []
    for path, entries in by_image.items():
        flagged = [e for e in entries if e.get("review_reasons")]
        if not flagged:
            continue
        reasons = {r for e in flagged for r in e["review_reasons"]}
        confs = [e["species_conf"] for e in flagged if e.get("species_conf") is not None]
        described = []
        for e in entries:
            if e["category"] == "blank":
                continue
            index = "recheck" if e["detection_index"] is None else e["detection_index"]
            text = f"{index}: {e['prediction']}"
            if e.get("frame_time") is not None:
                text += f" at {e['frame_time']:g}s"
            if e.get("species_conf") is not None:
                text += f" {e['species_conf']:.2f}"
            if e.get("prediction_raw") and e["prediction_raw"] != e["prediction"]:
                text += f" (top guess: {e['prediction_raw']})"
            if e.get("review_reasons"):
                text += f" [{', '.join(e['review_reasons'])}]"
            described.append(text)
        first = entries[0]
        rows.append({
            "file": os.path.relpath(path, source_root),
            "deployment_id": first.get("deployment_id"),
            "timestamp": first.get("timestamp"),
            "event_id": first.get("event_id"),
            "reasons": ";".join(r for r in REASONS if r in reasons),
            "lowest_conf": round(min(confs), 4) if confs else None,
            "detections": "; ".join(described),
        })
    rows.sort(key=lambda r: (r["deployment_id"] or "", r["timestamp"] or "~", r["file"]))
    return rows


def write_review_csv(results, path, source_root):
    """Write review.csv; returns the number of rows (images)."""
    rows = review_rows(results, source_root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
