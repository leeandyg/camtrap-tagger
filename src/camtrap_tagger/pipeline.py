"""detect → classify → per-animal decisions → whole-image blank recheck → video summaries → merge."""

import os
import time

from speciesnet import DEFAULT_MODEL
from speciesnet.classifier import SpeciesNetClassifier
from speciesnet.constants import NUM_CLASSIFICATIONS_FOR_ENSEMBLE
from speciesnet.geolocation import find_admin1_region
from speciesnet.utils import IMG_EXTENSIONS, build_classifier_result, load_rgb_image
from tqdm import tqdm

from camtrap_tagger.backends import CropBackend, UpstreamBackend
from camtrap_tagger.events import DEFAULT_GAP_MINUTES
from camtrap_tagger.metadata import VIDEO_MEDIATYPES, assign_deployments, read_timestamp
from camtrap_tagger.results import build_results, parse_ensemble_prediction
from camtrap_tagger.review import DEFAULT_REVIEW_BELOW
from camtrap_tagger.video import DEFAULT_VIDEO_FPS, tag_videos

WHOLE_IMAGE_MODEL = "kaggle:google/speciesnet/pyTorch/v4.0.3b/1"
DEFAULT_THRESHOLD = 0.1
# On held-out data, unfiltered recheck results mostly flagged truly empty frames as "unknown"
# or as low-confidence animals; only confident labels are kept.
RECHECK_MIN_SCORE = 0.8


def list_images(folder):
    """All images under folder (recursive), sorted."""
    return _list_files(folder, {f".{ext.lower()}" for ext in IMG_EXTENSIONS})


def list_videos(folder):
    """All videos under folder (recursive), sorted."""
    return _list_files(folder, VIDEO_MEDIATYPES)


def _list_files(folder, extensions):
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
        crops_dir=None, deployments=None, file_time_fallback=False,
        event_gap=DEFAULT_GAP_MINUTES, review_below=DEFAULT_REVIEW_BELOW,
        video_fps=DEFAULT_VIDEO_FPS):
    """Tag every image and video under `source`.

    `deployments` is an optional list from metadata.load_deployment_sheet; without it each
    folder is its own deployment. `file_time_fallback` uses file modified times for images
    without an EXIF capture time. `event_gap` is the minutes between images that starts a new
    event. Labels below `review_below` confidence are flagged for review. Videos are sampled
    at `video_fps` frames per second and summarized as one item each; the recheck skips them.

    Returns (results, predictions): the flat per-detection results dict, and the raw
    SpeciesNet predictions it was built from (per frame for videos).
    """
    filepaths = list_images(source)
    videos = list_videos(source)
    if not filepaths and not videos:
        raise ValueError(f"No images or videos found in {source}")

    assigned, all_deployments, warnings = assign_deployments(filepaths + videos, source, deployments)
    by_id = {d.deployment_id: d for d in all_deployments}
    image_info = {}
    for path in filepaths + videos:
        timestamp, timestamp_source = read_timestamp(path, file_time_fallback)
        image_info[path] = {
            "deployment_id": assigned[path],
            "timestamp": timestamp,
            "timestamp_source": timestamp_source,
            "outside_deployment": not by_id[assigned[path]].contains(timestamp),
        }

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
    predictions = tagger.predict(filepaths, country, admin1_region, threshold) if filepaths else []
    timing["detect_and_classify"] = time.perf_counter() - start

    if recheck:
        start = time.perf_counter()
        timing["rechecked_images"] = recheck_blanks(
            predictions, tagger.ensemble, whole_image_model, country, admin1_region, threshold)
        timing["recheck"] = time.perf_counter() - start

    start = time.perf_counter()
    summaries, frame_predictions, video_info = tag_videos(
        videos, tagger, source, video_fps, country, admin1_region, threshold)
    if videos:
        timing["videos"] = time.perf_counter() - start

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
    metadata["event_gap_minutes"] = event_gap
    metadata["review_below"] = review_below
    metadata["video_fps"] = video_fps if videos else None
    results = build_results(predictions + summaries, metadata, threshold, image_info, all_deployments,
                            warnings, event_gap, review_below, video_info)
    return results, predictions + frame_predictions
