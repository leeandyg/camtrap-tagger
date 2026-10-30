import csv
import json
import os
import struct
from datetime import datetime

import cv2
import numpy as np
import pytest

from camtrap_tagger.camtrapdp import parse_timezone, write_camtrap_dp
from camtrap_tagger.metadata import read_timestamp, read_video_creation_time
from camtrap_tagger.pipeline import list_images, list_videos
from camtrap_tagger.results import build_results, write_csv
from camtrap_tagger.review import review_rows
from camtrap_tagger.timelapse import build_recognition_file
from camtrap_tagger.video import extract_frames, read_frame, summarize_video, tag_videos
from camtrap_tagger.visualize import visualize_results

DEER = "u1;mammalia;artiodactyla;cervidae;odocoileus;virginianus;white-tailed deer"
ELK = "u2;mammalia;artiodactyla;cervidae;cervus;canadensis;elk"
CERVIDAE = "u3;mammalia;artiodactyla;cervidae;;;cervidae family"
HUMAN = "990ae9dd-7a59-4344-afcb-1b7b21368000;mammalia;primates;hominidae;homo;sapiens;human"
BLANK = "f1856211-cfb7-4a5b-9158-c0f72fd09ee6;;;;;;blank"


def detection(label, score=0.9, conf=0.8, detector="animal", bbox=(0.1, 0.2, 0.3, 0.4)):
    category = {"animal": "1", "human": "2", "vehicle": "3"}[detector]
    return {"category": category, "label": detector, "conf": conf, "bbox": list(bbox),
            "raw_classifications": {"classes": [label], "scores": [score]},
            "processed_classification": {"class": label, "score": score, "source": "classifier"}}


def summarize(frames, frame_rate=30.0, threshold=0.1):
    predictions = {n: {"filepath": f"f{n}.jpg", "detections": dets} for n, dets in frames.items()}
    return summarize_video("/v/clip.mp4", frame_rate, predictions, threshold)


def make_video(path, frames=60, fps=30.0, size=(64, 48)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert writer.isOpened()
    for i in range(frames):  # brightness encodes the frame number
        writer.write(np.full((size[1], size[0], 3), i * 4, dtype=np.uint8))
    writer.release()


class FakeBackend:
    """Stands in for a model: a detection on every frame, nothing to load."""

    def __init__(self, label=DEER):
        self.label = label
        self.calls = []

    def predict(self, filepaths, country=None, admin1_region=None, threshold=0.1):
        self.calls.append(list(filepaths))
        assert all(os.path.exists(p) for p in filepaths)
        return [{"filepath": p, "detections": [detection(self.label)]} for p in filepaths]


# Summarizing frames

def test_count_is_the_most_seen_in_one_frame_not_the_total():
    summary = summarize({0: [detection(DEER)], 30: [detection(DEER), detection(DEER)], 60: [detection(DEER)]})
    assert len(summary["detections"]) == 2
    first = summary["detections"][0]
    assert first["frame_number"] == 30 and first["frame_time"] == 1.0
    assert (first["first_seen"], first["last_seen"]) == (0.0, 2.0)
    assert (first["frames_seen"], first["frames_sampled"]) == (3, 3)
    assert first["frames_disagree"] is False


def test_best_frame_ties_go_to_the_most_confident():
    summary = summarize({0: [detection(DEER, score=0.7)], 30: [detection(DEER, score=0.95)]})
    assert [d["frame_number"] for d in summary["detections"]] == [30]


def test_broader_labels_fold_into_the_specific_one():
    summary = summarize({0: [detection(CERVIDAE, score=0.99), detection(CERVIDAE, score=0.98)],
                         30: [detection(DEER, score=0.9)]})
    # Two boxes from the frame showing most, both named by the specific label.
    assert [d["frame_number"] for d in summary["detections"]] == [0, 0]
    assert {d["processed_classification"]["class"] for d in summary["detections"]} == {DEER}
    assert not summary["detections"][0]["frames_disagree"]


def test_two_species_are_both_kept_and_flagged():
    results = build_results(
        [summarize({0: [detection(DEER)], 30: [detection(ELK)], 60: [detection(HUMAN, detector="human")]})],
        {"model": "m"}, 0.1)
    by_label = {e["prediction"]: e for e in results["predictions"]}
    assert set(by_label) == {"white-tailed deer", "elk", "human"}
    assert by_label["elk"]["review_reasons"] == ["frames_disagree"]
    assert by_label["human"]["review_reasons"] == []


def test_empty_and_unreadable_videos():
    empty = summarize({0: [], 30: [detection(BLANK)], 60: [detection(DEER, conf=0.05)]})
    assert empty["detections"] == []
    results = build_results([empty], {"model": "m"}, 0.1)
    assert results["predictions"][0]["category"] == "blank"
    failed = summarize_video("/v/clip.mp4", 30.0, {0: {"filepath": "f", "failures": ["DETECTOR"]}}, 0.1)
    assert failed == {"filepath": "/v/clip.mp4", "failures": ["VIDEO"]}
    assert summarize_video("/v/clip.mp4", None, {}, 0.1)["failures"] == ["VIDEO"]


def test_one_video_is_one_item_in_results_and_events():
    prediction = summarize({0: [detection(DEER)], 30: [detection(DEER)]})
    info = {"/v/clip.mp4": {"deployment_id": "d", "timestamp": "2024-01-01T10:00:00",
                            "timestamp_source": "video_metadata", "outside_deployment": False}}
    results = build_results([prediction], {"model": "m"}, 0.1, info)
    assert results["summary"]["videos"] == 1
    assert results["summary"]["species"]["white-tailed deer"] == {"detections": 1, "images": 1, "events": 1}
    assert results["events"][0]["observations"][0]["count"] == 1


# Reading real video files

def test_extract_samples_at_the_requested_rate(tmp_path):
    make_video(tmp_path / "a.mp4", frames=60, fps=30.0)
    rate, frames = extract_frames(str(tmp_path / "a.mp4"), str(tmp_path / "f"), 1.0)
    assert rate == 30.0 and sorted(frames) == [0, 30]
    assert all(os.path.exists(p) for p in frames.values())
    _, every = extract_frames(str(tmp_path / "a.mp4"), str(tmp_path / "g"), 1000.0)
    assert len(every) == 60


def test_read_frame_returns_the_same_frame_that_was_sampled(tmp_path):
    make_video(tmp_path / "a.mp4")
    frame = read_frame(str(tmp_path / "a.mp4"), 30)
    assert frame.size == (64, 48)
    assert np.asarray(frame).mean() > np.asarray(read_frame(str(tmp_path / "a.mp4"), 0)).mean() + 50
    with pytest.raises(OSError):
        read_frame(str(tmp_path / "a.mp4"), 1000)


def test_corrupt_video_is_reported_not_fatal(tmp_path):
    (tmp_path / "bad.mp4").write_bytes(b"not a video")
    assert extract_frames(str(tmp_path / "bad.mp4"), str(tmp_path / "f")) == (None, {})
    summaries, raw, info = tag_videos([str(tmp_path / "bad.mp4")], FakeBackend(), str(tmp_path))
    assert summaries[0]["failures"] == ["VIDEO"] and raw == [] and info == {}


def test_tag_videos_end_to_end(tmp_path):
    (tmp_path / "site").mkdir()
    video = tmp_path / "site" / "a.mp4"
    make_video(video)
    backend = FakeBackend()
    summaries, raw, info = tag_videos([str(video)], backend, str(tmp_path))
    assert [d["frame_number"] for d in summaries[0]["detections"]] == [0]
    assert info[str(video)] == {"frame_rate": 30.0, "frames_processed": [0, 30]}
    assert [p["filepath"] for p in raw] == [f"{video}#frame=0", f"{video}#frame=30"]
    assert not any(os.path.exists(p) for p in backend.calls[0])  # temporary frames cleaned up


def test_frames_are_chunked_so_they_do_not_pile_up_on_disk(tmp_path, monkeypatch):
    monkeypatch.setattr("camtrap_tagger.video.FRAME_CHUNK", 2)
    videos = [tmp_path / f"{n}.mp4" for n in range(3)]
    for video in videos:
        make_video(video)
    backend = FakeBackend()
    summaries, _, _ = tag_videos(list(map(str, videos)), backend, str(tmp_path))
    assert len(backend.calls) == 3 and len(summaries) == 3


def test_listing_separates_images_and_videos(tmp_path):
    for name in ["a.jpg", "b.MP4", "c.avi", ".hidden.mp4", "notes.txt"]:
        (tmp_path / name).touch()
    assert [os.path.basename(p) for p in list_videos(tmp_path)] == ["b.MP4", "c.avi"]
    assert [os.path.basename(p) for p in list_images(tmp_path)] == ["a.jpg"]


# Capture time

def mp4_with_creation_time(path, seconds, version=0):
    ftyp = struct.pack(">I", 16) + b"ftypisom" + b"\0" * 4
    mdat = struct.pack(">I", 12) + b"mdat" + b"\0" * 4  # before moov, as most cameras write it
    width = ">Q" if version else ">I"
    mvhd_body = bytes([version, 0, 0, 0]) + struct.pack(width, seconds)
    mvhd = struct.pack(">I", 8 + len(mvhd_body)) + b"mvhd" + mvhd_body
    moov = struct.pack(">I", 8 + len(mvhd)) + b"moov" + mvhd
    path.write_bytes(ftyp + mdat + moov)


@pytest.mark.parametrize("version", [0, 1])
def test_creation_time_from_mp4_header(tmp_path, version):
    seconds = int((datetime(2024, 5, 1, 12, 30) - datetime(1904, 1, 1)).total_seconds())
    mp4_with_creation_time(tmp_path / "a.mp4", seconds, version)
    assert read_video_creation_time(str(tmp_path / "a.mp4")) == "2024-05-01T12:30:00"
    assert read_timestamp(str(tmp_path / "a.mp4")) == ("2024-05-01T12:30:00", "video_metadata")


def test_unset_or_missing_creation_time(tmp_path):
    mp4_with_creation_time(tmp_path / "zero.mp4", 0)
    assert read_timestamp(str(tmp_path / "zero.mp4")) == (None, None)
    (tmp_path / "junk.mp4").write_bytes(b"\xff" * 100)
    assert read_video_creation_time(str(tmp_path / "junk.mp4")) is None
    (tmp_path / "a.avi").write_bytes(b"RIFF")
    assert read_timestamp(str(tmp_path / "a.avi")) == (None, None)
    assert read_timestamp(str(tmp_path / "a.avi"), file_time_fallback=True)[1] == "file_modified"


# Outputs

@pytest.fixture
def video_results(tmp_path):
    video = tmp_path / "a.mp4"
    make_video(video)
    path = str(video)
    prediction = summarize_video(path, 30.0, {
        0: {"filepath": "f0", "detections": [detection(DEER)]},
        30: {"filepath": "f1", "detections": [detection(DEER, score=0.95)]},
    }, 0.1)
    return build_results([prediction], {"model": "m", "threshold": 0.1}, 0.1,
                         videos={path: {"frame_rate": 30.0, "frames_processed": [0, 30]}}), tmp_path


def test_timelapse_video_shape(video_results):
    results, root = video_results
    [image] = build_recognition_file(results, str(root))["images"]
    assert image["file"] == "a.mp4" and image["frame_rate"] == 30.0
    assert image["frames_processed"] == [0, 30]
    assert image["detections"][0]["frame_number"] == 30


def test_timelapse_unreadable_video_still_has_the_required_fields(tmp_path):
    results = build_results([{"filepath": str(tmp_path / "bad.mp4"), "failures": ["VIDEO"]}],
                            {"model": "m", "threshold": 0.1}, 0.1)
    [image] = build_recognition_file(results, str(tmp_path))["images"]
    assert image["detections"] is None and image["failure"]
    assert image["frame_rate"] == 0 and image["frames_processed"] == []


def test_camtrap_dp_video_has_times_on_screen_and_no_box(video_results, tmp_path):
    results, root = video_results
    path = str(root / "a.mp4")
    for e in results["predictions"]:
        e.update(deployment_id="d", timestamp="2024-07-01T10:00:00")
    results["deployments"] = [{"deploymentID": "d", "latitude": 44.1, "longitude": -123.4,
                               "start": "2024-06-01T00:00:00", "end": "2024-08-01T00:00:00",
                               "locationName": None}]
    write_camtrap_dp(results, str(tmp_path / "dp"), str(root), parse_timezone("-07:00"), "Test")
    media = next(csv.DictReader((tmp_path / "dp" / "media.csv").open()))
    assert media["fileMediatype"] == "video/mp4" and media["filePath"] == "a.mp4"
    assert os.path.basename(path) == media["fileName"]
    observation = next(csv.DictReader((tmp_path / "dp" / "observations.csv").open()))
    assert observation["eventStart"] == "2024-07-01T10:00:00-07:00"
    assert observation["eventEnd"] == "2024-07-01T10:00:01-07:00"
    assert observation["bboxX"] == "" and observation["count"] == "1"


def test_visualization_draws_on_the_frame(video_results):
    results, root = video_results
    assert visualize_results(results, str(root), str(root / "viz")) == 1
    assert os.path.exists(root / "viz" / "a.mp4_frame000030.jpg")


def test_csv_and_review_list_show_the_time(video_results):
    results, root = video_results
    write_csv(results, str(root / "r.csv"), str(root))
    header = (root / "r.csv").read_text().splitlines()[0]
    assert "frame_time" in header and "frames_seen" in header
    results["predictions"][0]["review_reasons"] = ["low_confidence"]
    [listed] = review_rows(results, str(root))
    assert listed["detections"].startswith("0: white-tailed deer at 1s 0.95")
    json.dumps(results)  # stays serializable
