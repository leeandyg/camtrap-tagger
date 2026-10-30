"""Sample a labeled evaluation set from a LILA camera-trap dataset.

Reads a COCO Camera Traps .json, keeps images whose every annotation has a bounding box
(or that are empty), samples them in three strata (multi-animal, single-animal, empty),
downloads the images and writes ground_truth.json with boxes normalized to [0, 1] and
labels mapped to Wildlife Insights GUIDs (the IDs SpeciesNet's labels use).

Example (ENA24):
    uv run python eval/prepare_lila.py \\
        --coco eval/data/meta/ena24.json \\
        --image-url-prefix https://storage.googleapis.com/public-datasets-lila/ena24/images/ \\
        --dataset-name ENA24 --mapping eval/data/meta/lila_wi_mapping_table.csv \\
        --out eval/data/ena24 --multi 100 --single 150 --empty 50
"""

import argparse
import csv
import json
import os
import random
import urllib.parse
import urllib.request
from collections import defaultdict

from PIL import Image

HUMAN_GUID = "990ae9dd-7a59-4344-afcb-1b7b21368000"
VEHICLE_GUID = "e2895ed5-780b-48f6-8a11-9e27cb594511"
BLANK_GUID = "f1856211-cfb7-4a5b-9158-c0f72fd09ee6"


def load_mapping(path, dataset_name):
    """LILA category name (lowercased) → (WI GUID, WI common name) for one dataset."""
    with open(path, newline="", encoding="utf-8") as f:
        return {
            normalize(row["lila_category_name"]): (row["wi_guid"], row["wi_common"])
            for row in csv.DictReader(f)
            if row["lila_dataset_name"] == dataset_name
        }


# LILA scientific names whose SpeciesNet taxonomy uses an older name.
SYNONYMS = {
    "neogale vison": "neovison vison",            # American mink, genus renamed
    "dryocopus pileatus": "hylatomus pileatus",   # pileated woodpecker, genus differs
    # Humboldt's flying squirrel was split from the northern flying squirrel in 2017;
    # SpeciesNet only has G. sabrinus, so that is the best answer it can give.
    "glaucomys oregonensis": "glaucomys sabrinus",
    "dryobates villosus": "leuconotopicus villosus",  # hairy woodpecker
    "neogale frenata": "mustela frenata",              # long-tailed weasel
    "astur gentilis": "accipiter gentilis",            # northern goshawk
    "canachites canadensis": "falcipennis canadensis",  # spruce grouse
    # American ermine, split from the stoat in 2021; SpeciesNet only has M. erminea.
    "mustela richardsonii": "mustela erminea",
}

# Labels at ranks SpeciesNet's taxonomy doesn't have, mapped to the nearest rank it does.
RANK_OVERRIDES = {
    ("tribe", "tamiini"): ("genus", "tamias"),  # "chipmunk"
}


def normalize(name):
    """Category names differ in case and spacing between files ("american crow" vs "americancrow")."""
    return name.lower().replace(" ", "")


def index_taxonomy(path):
    """SpeciesNet taxonomy file → {(rank, name): (guid, common name)} for the deepest set rank.

    Species are keyed as "genus epithet", matching LILA's scientific names.
    """
    index = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(";")
            if len(parts) < 7:
                continue
            guid, ranks, common = parts[0], parts[1:6], parts[6]
            if ranks[4]:
                key = ("species", f"{ranks[3]} {ranks[4]}")
            else:
                depth = max((i for i, r in enumerate(ranks) if r), default=None)
                if depth is None:
                    continue
                key = (["class", "order", "family", "genus"][depth], ranks[depth])
            index.setdefault(key, (guid, common))
    return index


def load_mapping_by_scientific_name(path, dataset_name, taxonomy_index):
    """Map categories via lila-taxonomy-mapping_release.csv scientific names → SpeciesNet GUIDs.

    For datasets newer than the LILA→WI mapping table. Subspecies map to their species;
    kingdom-level labels (e.g. "invertebrate") are left unmapped, so their images are skipped.
    """
    mapping = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["dataset_name"] != dataset_name:
                continue
            name, level, sci = normalize(row["query"]), row["taxonomy_level"], row["scientific_name"]
            if name in ("empty", "blank"):
                mapping[name] = (BLANK_GUID, "blank")
                continue
            if level == "subspecies":
                level, sci = "species", " ".join(sci.split()[:2])
            level, sci = RANK_OVERRIDES.get((level, sci), (level, sci))
            match = taxonomy_index.get((level, SYNONYMS.get(sci, sci)))
            if match:
                mapping[name] = match
    return mapping


def object_category(guid):
    return {HUMAN_GUID: "human", VEHICLE_GUID: "vehicle"}.get(guid, "animal")


def build_candidates(coco, mapping, image_level=False):
    """Images whose labels can be scored, as ground-truth dicts.

    By default every animal needs a box (per-detection scoring). With image_level, labels
    without boxes are kept with "bbox": None, one object per distinct taxon in the image.
    """
    categories = {c["id"]: c["name"] for c in coco["categories"]}
    annotations = defaultdict(list)
    for a in coco["annotations"]:
        annotations[a["image_id"]].append(a)

    candidates = []
    unmapped = set()
    for img in coco["images"]:
        # Sequence-level labels may not apply to every frame of the burst.
        if img.get("sequence_level_annotation"):
            continue
        objects, usable = [], True
        for a in annotations[img["id"]]:
            name = categories[a["category_id"]]
            guid, common = mapping.get(normalize(name), (None, None))
            if guid is None:
                unmapped.add(name)
                usable = False
                break
            if guid == BLANK_GUID:
                continue
            if image_level:
                if all(o["wi_guid"] != guid for o in objects):
                    objects.append({"bbox": None, "wi_guid": guid, "label": common,
                                    "category": object_category(guid)})
                continue
            if "bbox" not in a or not img.get("width"):
                usable = False
                break
            x, y, w, h = a["bbox"]
            objects.append({
                "bbox": [x / img["width"], y / img["height"], w / img["width"], h / img["height"]],
                "wi_guid": guid,
                "label": common,
                "category": object_category(guid),
            })
        if usable:
            candidates.append({"file": img["file_name"], "objects": objects})
    if unmapped:
        print(f"Skipped images with unmapped categories: {sorted(unmapped)}")
    return candidates


def stratify(candidates, n_multi, n_single, n_empty, seed):
    def animals(c):
        return sum(o["category"] == "animal" for o in c["objects"])

    strata = {
        "multi": [c for c in candidates if animals(c) >= 2],
        "single": [c for c in candidates if animals(c) == 1],
        "empty": [c for c in candidates if not c["objects"]],
    }
    rng = random.Random(seed)
    sample = []
    for name, n in [("multi", n_multi), ("single", n_single), ("empty", n_empty)]:
        pool = strata[name]
        picked = rng.sample(pool, min(n, len(pool)))
        print(f"{name}: {len(picked)} of {len(pool)} available")
        sample.extend({**c, "stratum": name} for c in picked)
    return sample


def download(sample, url_prefix, images_dir):
    kept = []
    for item in sample:
        dest = os.path.join(images_dir, item["file"])
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        try:
            if not os.path.exists(dest):
                urllib.request.urlretrieve(url_prefix + urllib.parse.quote(item["file"]), dest)
            Image.open(dest).verify()
            kept.append(item)
        except Exception as e:  # skip broken downloads rather than abort the sample
            print(f"  Skipping {item['file']}: {e}")
    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--coco", required=True, help="COCO Camera Traps .json")
    parser.add_argument("--image-url-prefix", required=True)
    parser.add_argument("--dataset-name", required=True, help="name as in the LILA→WI mapping table")
    mapping_source = parser.add_mutually_exclusive_group(required=True)
    mapping_source.add_argument("--mapping", help="lila_wi_mapping_table.csv")
    mapping_source.add_argument("--taxonomy-csv",
                                help="lila-taxonomy-mapping_release.csv (for datasets not in the WI table)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--multi", type=int, default=100)
    parser.add_argument("--single", type=int, default=150)
    parser.add_argument("--empty", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-level", action="store_true",
                        help="dataset has no boxes: score labels per image (multi = 2+ species)")
    args = parser.parse_args()

    with open(args.coco, encoding="utf-8") as f:
        coco = json.load(f)
    if args.mapping:
        mapping = load_mapping(args.mapping, args.dataset_name)
    else:
        from speciesnet import DEFAULT_MODEL
        from speciesnet.utils import ModelInfo
        taxonomy_index = index_taxonomy(ModelInfo(DEFAULT_MODEL).taxonomy)
        mapping = load_mapping_by_scientific_name(args.taxonomy_csv, args.dataset_name, taxonomy_index)
    if not mapping:
        raise SystemExit(f"No mapping rows for dataset '{args.dataset_name}'")

    candidates = build_candidates(coco, mapping, args.image_level)
    del coco
    sample = stratify(candidates, args.multi, args.single, args.empty, args.seed)
    sample = download(sample, args.image_url_prefix, os.path.join(args.out, "images"))

    with open(os.path.join(args.out, "ground_truth.json"), "w") as f:
        json.dump({
            "dataset": args.dataset_name,
            "source": args.image_url_prefix,
            "seed": args.seed,
            "image_level": args.image_level,
            "images": sample,
        }, f, indent=1)
    print(f"Wrote {len(sample)} images to {args.out}")


if __name__ == "__main__":
    main()
