"""Group detections into independent events (visits).

An event is a run of images from one deployment, each taken within `gap` of the previous
one, that contain something other than blank. Each event gets one observation per label,
with the count being the most detections of that label in any single image (so a deer seen
in five photos of a burst counts once).

SpeciesNet can label the same animal differently across a burst ("white-tailed deer" in one
frame, "cervidae family" or "unknown" in the next). Within an event, a broader label is
folded into the more specific one when there is exactly one specific candidate; otherwise it
stays as its own observation.
"""

from collections import Counter, defaultdict
from datetime import datetime, timedelta

DEFAULT_GAP_MINUTES = 30
GENERIC_LABELS = {"animal", "unknown"}


def label_ranks(entry):
    """[class, order, family, genus, species] of an entry's final label (empty if none)."""
    parts = (entry.get("ensemble_prediction_raw") or "").split(";")
    return (parts[1:6] + [""] * 5)[:5]


def is_ancestor(coarse, fine):
    """True if `coarse` ranks are a strict prefix of `fine` ranks."""
    depth = sum(1 for r in coarse if r)
    return 0 < depth < sum(1 for r in fine if r) and coarse[:depth] == fine[:depth]


def fold_labels(ranks_by_label):
    """Map each animal label to the label it should be counted as within one event."""
    specific = [n for n, r in ranks_by_label.items() if n not in GENERIC_LABELS and any(r)]
    folded = {}
    for name, ranks in ranks_by_label.items():
        if name in GENERIC_LABELS:
            candidates = specific
        else:
            candidates = [n for n in specific if is_ancestor(ranks, ranks_by_label[n])]
        leaves = [c for c in candidates
                  if not any(is_ancestor(ranks_by_label[c], ranks_by_label[o]) for o in candidates)]
        folded[name] = leaves[0] if len(leaves) == 1 else name
    return folded


def build_events(entries, gap_minutes=DEFAULT_GAP_MINUTES):
    """Group entries into events.

    Returns (events, event ID per filepath). Each event has event_id, deployment_id, start,
    end, images and observations [{label, category, count, confidence}].
    """
    images = defaultdict(list)
    for e in entries:
        images[e["filepath"]].append(e)

    by_deployment = defaultdict(list)
    for path, image_entries in images.items():
        first = image_entries[0]
        found = [e for e in image_entries if e["category"] not in ("blank", "error")]
        if found and first.get("timestamp"):
            by_deployment[first.get("deployment_id")].append(
                (datetime.fromisoformat(first["timestamp"]), path, found))

    gap = timedelta(minutes=gap_minutes)
    events, event_of = [], {}
    for deployment_id in sorted(by_deployment, key=str):
        runs = []
        for taken, path, found in sorted(by_deployment[deployment_id], key=lambda x: (x[0], x[1])):
            if runs and taken - runs[-1][-1][0] <= gap:
                runs[-1].append((taken, path, found))
            else:
                runs.append([(taken, path, found)])

        for n, run in enumerate(runs, start=1):
            event_id = f"{deployment_id}-{n:04d}"
            events.append({
                "event_id": event_id,
                "deployment_id": deployment_id,
                "start": run[0][0].isoformat(),
                "end": run[-1][0].isoformat(),
                "images": len(run),
                "observations": event_observations([found for _, _, found in run]),
            })
            for _, path, _ in run:
                event_of[path] = event_id
    return events, event_of


def event_observations(images):
    """One observation per (folded) label: max count in a single image, best confidence."""
    ranks = {e["prediction"]: label_ranks(e) for found in images for e in found
             if e["category"] in ("animal", "unknown")}
    folded = fold_labels(ranks)

    def label(e):
        if e["category"] in ("animal", "unknown"):
            return folded.get(e["prediction"], e["prediction"])
        return e["category"]

    counts, confidence, category = Counter(), {}, {}
    for found in images:
        for name, n in Counter(label(e) for e in found).items():
            counts[name] = max(counts[name], n)
        for e in found:
            name = label(e)
            conf = e.get("species_conf") if e.get("species_conf") is not None else e.get("detector_conf")
            confidence[name] = max(confidence.get(name) or 0.0, conf or 0.0)
            # A folded label takes the category of the specific label it became.
            if name == e["prediction"] or name not in category:
                category[name] = e["category"]
    return [{"label": name, "category": category[name], "count": counts[name],
             "confidence": round(confidence[name], 4)}
            for name in sorted(counts, key=lambda n: (-counts[n], n))]
