"""Score camtrap-tagger results against LILA ground truth (see prepare_lila.py).

Each ground-truth animal box is matched to at most one predicted box (IoU >= 0.5), then
its label is graded against SpeciesNet's taxonomy:

  exact      same taxon as the ground truth, or a subspecies/domestic form of it
  coarser    a correct ancestor (e.g. "rabbit and hare family" for an eastern cottontail)
  animal     only "animal" (the generic fallback)
  finer      more specific than a coarse ground-truth label (e.g. "american crow" for "bird")
  wrong      a different taxon, or human/vehicle
  unknown    SpeciesNet declined to label it
  blank      the box was called blank
  missed     no predicted box; "recovered" if the whole-image recheck found an animal instead
"""

from collections import Counter

IOU_THRESHOLD = 0.5
ANIMAL_OUTCOMES = ["exact", "coarser", "animal", "finer", "wrong", "unknown", "blank",
                   "missed", "recovered"]


def load_taxonomy(path):
    """SpeciesNet taxonomy file → {guid: [class, order, family, genus, species]}."""
    taxa = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(";")
            if len(parts) >= 6:
                taxa[parts[0]] = parts[1:6]
    return taxa


def label_ranks(label):
    """(guid, 5 ranks) from a full SpeciesNet label string."""
    parts = (label or "").split(";")
    return parts[0], (parts[1:6] + [""] * 5)[:5]


def is_ancestor(coarse, fine):
    """True if `coarse` ranks are a strict prefix of `fine` ranks (empty ranks = not set)."""
    depth = sum(1 for r in coarse if r)
    return 0 < depth < sum(1 for r in fine if r) and coarse[:depth] == fine[:depth]


def is_subspecies(ranks, truth):
    """True if `ranks` is a subspecies or domestic form of the species in `truth`.

    E.g. domestic chicken (gallus / "gallus domesticus") for red junglefowl (gallus / "gallus"),
    which is how LILA maps ENA24's "Chicken" label.
    """
    return bool(truth[4]) and ranks[:4] == truth[:4] and ranks[4].startswith(truth[4] + " ")


def grade(entry, truth_guid, taxonomy):
    """Grade one predicted entry against a ground-truth animal."""
    category = entry["category"]
    if category in ("human", "vehicle"):
        return "wrong"
    if category in ("blank", "unknown"):
        return category
    guid, ranks = label_ranks(entry.get("ensemble_prediction_raw"))
    truth = taxonomy.get(truth_guid)
    if guid == truth_guid or (truth and is_subspecies(ranks, truth)):
        return "exact"
    if entry["prediction"] == "animal" or not any(ranks):
        return "animal"
    if truth and is_ancestor(ranks, truth):
        return "coarser"
    if truth and is_ancestor(truth, ranks):
        return "finer"
    return "wrong"


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def match(truths, preds):
    """Greedy one-to-one matching by IoU. Returns {truth_index: pred_index}."""
    pairs = sorted(
        ((iou(t["bbox"], p["bbox"]), ti, pi)
         for ti, t in enumerate(truths) for pi, p in enumerate(preds)),
        reverse=True,
    )
    matched, used = {}, set()
    for score, ti, pi in pairs:
        if score < IOU_THRESHOLD:
            break
        if ti not in matched and pi not in used:
            matched[ti] = pi
            used.add(pi)
    return matched


def score_image(truth, entries, taxonomy):
    """Per-image outcomes: animal grades, human/vehicle results, extra boxes, image flags."""
    boxes = [e for e in entries if e.get("bbox")]
    # The whole-image recheck only runs when nothing was detected, and finds at most one animal.
    recheck_found_animal = not boxes and any(e.get("note") == "detector_missed" for e in entries)
    matched = match(truth["objects"], boxes)

    animals, others = Counter(), Counter()
    for ti, obj in enumerate(truth["objects"]):
        pred = boxes[matched[ti]] if ti in matched else None
        if obj["category"] == "animal":
            if pred:
                animals[grade(pred, obj["wi_guid"], taxonomy)] += 1
            elif recheck_found_animal and not animals["recovered"]:
                animals["recovered"] += 1
            else:
                animals["missed"] += 1
        else:
            others[f"{obj['category']}_{'found' if pred and pred['category'] == obj['category'] else 'missed'}"] += 1

    extra = Counter(boxes[pi]["category"] for pi in set(range(len(boxes))) - set(matched.values())
                    if boxes[pi]["category"] != "blank")

    n_truth_animals = sum(o["category"] == "animal" for o in truth["objects"])
    n_pred_animals = sum(e["category"] == "animal" for e in entries)
    good = animals["exact"] + animals["coarser"] + animals["finer"]
    return {
        "animals": animals,
        "others": others,
        "extra": extra,
        "empty_correct": (not truth["objects"]) and all(e["category"] == "blank" for e in entries),
        "count_correct": n_pred_animals == n_truth_animals,
        "all_animals_correct": n_truth_animals > 0 and good == n_truth_animals
                               and n_pred_animals == n_truth_animals,
    }


def score(ground_truth, results, taxonomy):
    """Aggregate metrics for one results dict against ground_truth.json contents."""
    by_file = {}
    for e in results["predictions"]:
        by_file.setdefault(e["filepath"], []).append(e)

    animals, others, extra = Counter(), Counter(), Counter()
    strata = Counter()
    stratum_hits = Counter()
    for truth in ground_truth["images"]:
        entries = next((v for k, v in by_file.items() if k.endswith("/" + truth["file"])), [])
        s = score_image(truth, entries, taxonomy)
        animals.update(s["animals"])
        others.update(s["others"])
        extra.update(s["extra"])
        stratum = truth["stratum"]
        strata[stratum] += 1
        if stratum == "empty":
            stratum_hits[stratum] += s["empty_correct"]
        else:
            stratum_hits[stratum] += s["all_animals_correct"]

    total = sum(animals.values())
    return {
        "images": sum(strata.values()),
        "truth_animals": total,
        "animal_outcomes": {k: animals[k] for k in ANIMAL_OUTCOMES},
        "species_exact_rate": animals["exact"] / total if total else None,
        "correct_any_level_rate": (animals["exact"] + animals["coarser"] + animals["finer"]) / total
                                  if total else None,
        "human_vehicle": dict(others),
        "extra_boxes": dict(extra),
        "image_rates": {k: stratum_hits[k] / strata[k] for k in strata},
    }
