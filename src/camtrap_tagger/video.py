"""Video: sample frames, run them as images, and reduce each video to one media item.

A deer on screen for 30 seconds is one animal, not 30 detections, and a reviewer opens the
video, not its frames. So each video becomes a single item in the results, like one image,
and events, species counts, the review list and the exports need no special handling.

The summary follows the same rules as events: one set of detections per label, taken from
the frame that shows the most of that label (so the count is the most seen at once), with a
broader label folded into the single more specific one. Species aren't voted on: frames that
give two species keep both and are flagged `frames_disagree` for review.
"""

import os
import tempfile
from collections import defaultdict

import cv2
from PIL import Image
from tqdm import tqdm

from camtrap_tagger.events import folded_label_fn
from camtrap_tagger.metadata import VIDEO_MEDIATYPES
from camtrap_tagger.results import detection_entry

DEFAULT_VIDEO_FPS = 1.0
FRAME_CHUNK = 1000  # frames kept on disk at once, so big collections don't fill it
JPEG_QUALITY = 95


def is_video(path):
    return os.path.splitext(path)[1].lower() in VIDEO_MEDIATYPES


def extract_frames(path, dest_dir, sample_fps=DEFAULT_VIDEO_FPS):
    """Save about sample_fps frames per second of video as JPEGs in dest_dir.

    Returns (frame rate, {frame number: JPEG path}); (None, {}) if the video can't be read.
    Frames are read in order instead of seeking, which is unreliable in some AVI files.
    """
    capture = cv2.VideoCapture(path)
    try:
        frame_rate = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or frame_rate <= 0:
            return None, {}
        os.makedirs(dest_dir, exist_ok=True)
        step = max(frame_rate / sample_fps, 1.0)
        frames, number, next_frame = {}, 0, 0.0
        while capture.grab():
            if number >= next_frame:
                ok, image = capture.retrieve()
                if ok:
                    # Not cv2.imwrite: yolov5, imported by SpeciesNet, replaces it with a
                    # version that takes no parameters.
                    encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1]
                    frames[number] = os.path.join(dest_dir, f"frame{number:06d}.jpg")
                    encoded.tofile(frames[number])
                next_frame += step
            number += 1
        return frame_rate, frames
    finally:
        capture.release()


def read_frame(path, frame_number):
    """One frame of a video as an RGB PIL image, decoded the same way as extract_frames."""
    capture = cv2.VideoCapture(path)
    try:
        for _ in range(frame_number):
            if not capture.grab():
                raise OSError(f"{path} has no frame {frame_number}")
        ok, image = capture.read()
        if not ok:
            raise OSError(f"{path} has no frame {frame_number}")
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()


def _confidence(entry):
    conf = entry.get("species_conf")
    return conf if conf is not None else (entry.get("detector_conf") or 0.0)


def summarize_video(path, frame_rate, frames, threshold):
    """One prediction for a video from its frame predictions ({frame number: prediction}).

    A box whose own label was folded takes the label of the best detection of the specific
    one. Boxes the classifier called blank are dropped, so a video with nothing else is blank.
    """
    readable = {n: p for n, p in frames.items() if p.get("detections") is not None}
    if not readable:
        return {"filepath": path, "failures": ["VIDEO"]}

    sightings = []  # (frame number, detection, its results entry)
    for number, pred in sorted(readable.items()):
        for i, det in enumerate(pred["detections"]):
            if det.get("category") != "0" and det.get("conf", 0) >= threshold:
                entry = detection_entry(path, i, det)
                if entry["category"] != "blank":
                    sightings.append((number, det, entry))

    label_of = folded_label_fn([entry for _, _, entry in sightings])
    by_label = defaultdict(list)
    for sighting in sightings:
        by_label[label_of(sighting[2])].append(sighting)
    animal_labels = {label_of(e) for _, _, e in sightings if e["category"] == "animal"}

    detections = []
    for label, seen in by_label.items():
        by_frame = defaultdict(list)
        for sighting in seen:
            by_frame[sighting[0]].append(sighting)
        best_frame = max(by_frame, key=lambda n: (
            len(by_frame[n]), max(_confidence(e) for _, _, e in by_frame[n]), -n))
        named = [s for s in seen if s[2]["prediction"] == label] or seen
        source = max(named, key=lambda s: _confidence(s[2]))[1]
        for _, det, entry in by_frame[best_frame]:
            labelled = det if entry["prediction"] == label else {
                **det,
                "raw_classifications": source.get("raw_classifications"),
                "processed_classification": source.get("processed_classification"),
            }
            detections.append({
                **labelled,
                "frame_number": best_frame,
                "frame_time": round(best_frame / frame_rate, 2),
                "first_seen": round(min(by_frame) / frame_rate, 2),
                "last_seen": round(max(by_frame) / frame_rate, 2),
                "frames_seen": len(by_frame),
                "frames_sampled": len(readable),
                "frames_disagree": entry["category"] == "animal" and len(animal_labels) > 1,
            })
    return {"filepath": path, "detections": detections}


def tag_videos(videos, tagger, source_root, sample_fps=DEFAULT_VIDEO_FPS, country=None,
               admin1_region=None, threshold=0.1):
    """Run videos through a backend, a chunk of frames at a time.

    Returns (one summary prediction per video, every frame's raw prediction with a filepath
    of "<video>#frame=<number>", and {video: frame_rate and frames_processed}).
    """
    summaries, raw, info = [], [], {}
    progress = tqdm(total=len(videos), desc="Reading videos")
    remaining = list(videos)
    while remaining:
        with tempfile.TemporaryDirectory(prefix="camtrap-tagger-frames-") as frames_dir:
            batch, frame_count = {}, 0
            while remaining and frame_count < FRAME_CHUNK:
                video = remaining.pop(0)
                dest = os.path.join(frames_dir, os.path.relpath(video, source_root))
                batch[video] = extract_frames(video, dest, sample_fps)
                frame_count += len(batch[video][1])
                progress.update()
            paths = [p for _, frames in batch.values() for p in frames.values()]
            tagger.source_root = frames_dir  # where the crop backend mirrors its crops from
            predicted = (tagger.predict(paths, country, admin1_region, threshold) if paths else [])
        by_path = {p["filepath"]: p for p in predicted}

        for video, (frame_rate, frames) in batch.items():
            frame_predictions = {n: by_path[p] for n, p in frames.items() if p in by_path}
            summaries.append(summarize_video(video, frame_rate, frame_predictions, threshold))
            if frames:
                info[video] = {"frame_rate": frame_rate, "frames_processed": sorted(frames)}
            raw.extend({**pred, "filepath": f"{video}#frame={n}"}
                       for n, pred in sorted(frame_predictions.items()))
    progress.close()
    return summaries, raw, info
