"""detect → classify → per-animal decisions → whole-image blank recheck → merge."""

import os
import time

from speciesnet import DEFAULT_MODEL
from speciesnet.classifier import SpeciesNetClassifier
from speciesnet.constants import NUM_CLASSIFICATIONS_FOR_ENSEMBLE
from speciesnet.geolocation import find_admin1_region
from speciesnet.utils import IMG_EXTENSIONS, build_classifier_result, load_rgb_image
from tqdm import tqdm

from camtrap_tagger.backends import CropBackend, UpstreamBackend
from camtrap_tagger.results import build_results, parse_ensemble_prediction

WHOLE_IMAGE_MODEL = "kaggle:google/speciesnet/pyTorch/v4.0.3b/1"
DEFAULT_THRESHOLD = 0.1
# On held-out data, unfiltered recheck results mostly flagged truly empty frames as "unknown"
# or as low-confidence animals; only confident labels are kept.
RECHECK_MIN_SCORE = 0.8


def list_images(folder):
    """All images under folder (recursive), sorted."""
    extensions = {f".{ext.lower()}" for ext in IMG_EXTENSIONS}
    found = []
    for root, _, files in os.walk(folder):
        for name in files:
            if not name.startswith(".") and os.path.splitext(name)[1].lower() in extensions:
                found.append(os.path.join(root, name))
    return sorted(found)


def accept_recheck(detection):
    """Whether a whole-image recheck result is trustworthy enough to report."""
    processed = detection.get("processed_classification") or {}
    parsed = parse_ensemble_prediction(processed.get("class"))
    if parsed["is_blank"]:
        return True
    return not parsed["is_unknown"] and processed.get("score", 0) >= RECHECK_MIN_SCORE


def recheck_blanks(predictions, ensemble, model_name, country=None, admin1_region=None,
                   threshold=DEFAULT_THRESHOLD):
    """Classify detector-empty images with the whole-image model.

    Images with no detection at or above the threshold get a `whole_image_recheck`
    detection holding the whole-image classification, ensembled as if the detector
    had found nothing, when `accept_recheck` trusts it. This can catch animals the
    detector missed.
    """
    targets = [
        p["filepath"] for p in predictions
        if p.get("detections") is not None
        and not any(d["conf"] >= threshold for d in p["detections"] if d["category"] != "0")
    ]
    if not targets:
        return 0

    classifier = SpeciesNetClassifier(model_name)
    classifier_results = {}
    for filepath in tqdm(targets, desc="Rechecking blanks"):
        inputs = classifier.preprocess(load_rgb_image(filepath))
        raw = classifier.predict(filepath, inputs, num_classes=NUM_CLASSIFICATIONS_FOR_ENSEMBLE)
        classifier_results[filepath] = build_classifier_result(raw, None, [], True)

    geolocation = {"country": country, "admin1_region": find_admin1_region(country, admin1_region)}
    combined = ensemble.combine(
        filepaths=targets,
        classifier_results=classifier_results,
        detector_results={fp: {"detections": []} for fp in targets},
        geolocation_results={fp: geolocation for fp in targets},
    )
    by_path = {p["filepath"]: p for p in predictions}
    for result in combined:
        if result.get("detections") and accept_recheck(result["detections"][0]):
            by_path[result["filepath"]]["whole_image_recheck"] = result["detections"][0]
    return len(targets)


def run(source, *, backend="upstream", model=DEFAULT_MODEL, whole_image_model=WHOLE_IMAGE_MODEL,
        country=None, admin1_region=None, threshold=DEFAULT_THRESHOLD, recheck=False,
        crops_dir=None):
    """Tag every image under `source`.

    Returns (results, predictions): the flat per-detection results dict, and the raw
    SpeciesNet predictions it was built from.
    """
    filepaths = list_images(source)
    if not filepaths:
        raise ValueError(f"No images found in {source}")

    timing = {}
    start = time.perf_counter()
    if backend == "crop":
        tagger = CropBackend(model, crops_dir=crops_dir, source_root=source)
    elif backend == "upstream":
        tagger = UpstreamBackend(model)
    else:
        raise ValueError(f"Unknown backend: {backend}")
    timing["load_models"] = time.perf_counter() - start

    start = time.perf_counter()
    predictions = tagger.predict(filepaths, country, admin1_region, threshold)
    timing["detect_and_classify"] = time.perf_counter() - start

    if recheck:
        start = time.perf_counter()
        timing["rechecked_images"] = recheck_blanks(
            predictions, tagger.ensemble, whole_image_model, country, admin1_region, threshold)
        timing["recheck"] = time.perf_counter() - start

    metadata = {
        "source_folder": os.path.abspath(source),
        "backend": backend,
        "model": model,
        "whole_image_model": whole_image_model if recheck else None,
        "threshold": threshold,
        "country": country,
        "admin1_region": admin1_region,
        "timing_seconds": timing,
    }
    return build_results(predictions, metadata, threshold), predictions
