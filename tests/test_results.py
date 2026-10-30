import csv
import os

import pytest

from camtrap_tagger.results import (
    build_results,
    derive_rollup_status,
    image_entries,
    parse_ensemble_prediction,
    parse_taxonomy_label,
    write_csv,
    write_species_csv,
)

DEER = "5c7ce479-8a45-40b3-ae21-7c97dfae22f5;mammalia;artiodactyla;cervidae;odocoileus;virginianus;white-tailed deer"
CERVIDAE = "abc;mammalia;artiodactyla;cervidae;;;cervidae family"
HUMAN = "990ae9dd-7a59-4344-afcb-1b7b21368000;mammalia;primates;hominidae;homo;sapiens;human"
BLANK = "f1856211-cfb7-4a5b-9158-c0f72fd09ee6;;;;;;blank"
ANIMAL = "1f689929-883d-4dae-958c-3d57ab5b6c16;;;;;;animal"
UNKNOWN = ("f2efdae9-efb8-48fb-8a91-eccf79ab4ffb;no cv result;no cv result;no cv result;"
           "no cv result;no cv result;no cv result")


def detection(label="animal", conf=0.9, bbox=(0.1, 0.1, 0.2, 0.2), raw=None, processed=None):
    category = {"animal": "1", "human": "2", "vehicle": "3"}[label]
    det = {"category": category, "label": label, "conf": conf, "bbox": list(bbox)}
    if raw:
        det["raw_classifications"] = {"classes": [raw], "scores": [0.95]}
    if processed:
        det["processed_classification"] = processed
    return det


def whole_image(raw, processed):
    return {
        "category": "0", "label": "whole_image_classification", "conf": 1.0,
        "bbox": [0.0, 0.0, 1.0, 1.0],
        "raw_classifications": {"classes": [raw], "scores": [0.9]},
        "processed_classification": processed,
    }


# ---- label parsing ----

def test_parse_taxonomy_label_full():
    parsed = parse_taxonomy_label(DEER)
    assert parsed["taxonomy_uuid"] == "5c7ce479-8a45-40b3-ae21-7c97dfae22f5"
    assert parsed["taxonomy_5part"] == "mammalia;artiodactyla;cervidae;odocoileus;virginianus"
    assert parsed["common_name"] == "white-tailed deer"


def test_parse_taxonomy_label_blank_and_missing():
    assert parse_taxonomy_label(BLANK)["common_name"] == "blank"
    assert parse_taxonomy_label(BLANK)["taxonomy_5part"] == ";;;;"
    assert parse_taxonomy_label(None)["common_name"] == "unknown"


def test_parse_taxonomy_label_short_label_is_padded():
    parsed = parse_taxonomy_label("mammalia;carnivora")
    assert parsed["taxonomy_uuid"] is None
    assert parsed["taxonomy_5part"] == "mammalia;carnivora;;;"
    assert parsed["common_name"] == "carnivora"


@pytest.mark.parametrize("label, name, is_blank, is_unknown", [
    (DEER, "white-tailed deer", False, False),
    (BLANK, "blank", True, False),
    (UNKNOWN, "unknown", False, True),
    ("blank", "blank", True, False),
    (None, "unknown", False, False),
])
def test_parse_ensemble_prediction(label, name, is_blank, is_unknown):
    parsed = parse_ensemble_prediction(label)
    assert (parsed["clean_name"], parsed["is_blank"], parsed["is_unknown"]) == (name, is_blank, is_unknown)


# ---- rollup status (source strings as produced by speciesnet 5.0.6) ----

@pytest.mark.parametrize("prediction, source, expected", [
    ("white-tailed deer", "classifier", "valid"),
    ("cervidae family", "classifier+rollup_to_family", "confidence_rollup"),
    ("cervidae family", "classifier+geofence+rollup_to_family", "geofence_rollup"),
    ("animal", "classifier+geofence+rollup_failed", "rolled_to_animal"),
    ("human", "detector", "detector"),
    ("blank", "classifier", "blank"),
    ("unknown", "classifier", "unknown"),
    ("mule deer", "", "rolled_up"),
])
def test_derive_rollup_status(prediction, source, expected):
    assert derive_rollup_status(prediction, source, "white-tailed deer") == expected


# ---- per-image entries ----

def test_classified_animal():
    pred = {"filepath": "a.jpg", "detections": [detection(
        raw=DEER, processed={"class": DEER, "score": 0.95, "source": "classifier"})]}
    [entry] = image_entries(pred, threshold=0.1)
    assert entry["category"] == "animal"
    assert entry["prediction"] == "white-tailed deer"
    assert entry["species_conf"] == 0.95
    assert entry["detector_conf"] == 0.9
    assert entry["rollup_status"] == "valid"
    assert "note" not in entry


def test_geofence_rollup_keeps_raw_top_class():
    pred = {"filepath": "a.jpg", "detections": [detection(
        raw=DEER, processed={"class": CERVIDAE, "score": 0.97, "source": "classifier+geofence+rollup_to_family"})]}
    [entry] = image_entries(pred, threshold=0.1)
    assert entry["prediction"] == "cervidae family"
    assert entry["prediction_raw"] == "white-tailed deer"
    assert entry["rollup_status"] == "geofence_rollup"


def test_unclassified_human_keeps_detector_label():
    # The crop backend only classifies animals.
    pred = {"filepath": "a.jpg", "detections": [detection("human", conf=0.8)]}
    [entry] = image_entries(pred, threshold=0.1)
    assert entry["category"] == "human"
    assert entry["prediction_source"] == "detector"
    assert entry["species_conf"] is None


def test_category_follows_ensemble_over_detector():
    # Upstream can relabel a low-confidence vehicle box as human.
    pred = {"filepath": "a.jpg", "detections": [detection(
        "vehicle", conf=0.49, raw=HUMAN, processed={"class": HUMAN, "score": 0.9, "source": "classifier"})]}
    [entry] = image_entries(pred, threshold=0.1)
    assert entry["category"] == "human"


def test_unknown_and_blank_animal_detections():
    pred = {"filepath": "a.jpg", "detections": [
        detection(raw=DEER, processed={"class": UNKNOWN, "score": 0.4, "source": "classifier"}),
        detection(raw=BLANK, processed={"class": BLANK, "score": 0.99, "source": "classifier"}),
    ]}
    categories = [e["category"] for e in image_entries(pred, threshold=0.1)]
    assert categories == ["unknown", "blank"]


def test_detections_below_threshold_are_dropped():
    pred = {"filepath": "a.jpg", "detections": [
        detection(conf=0.05),
        detection(conf=0.5, raw=DEER, processed={"class": DEER, "score": 0.9, "source": "classifier"}),
    ]}
    entries = image_entries(pred, threshold=0.1)
    assert [e["detection_index"] for e in entries] == [1]


def test_detector_missed_animal_found_by_recheck():
    pred = {"filepath": "a.jpg", "detections": [detection(conf=0.02)],
            "whole_image_recheck": whole_image(DEER, {"class": DEER, "score": 0.88, "source": "classifier"})}
    [entry] = image_entries(pred, threshold=0.1)
    assert entry["category"] == "animal"
    assert entry["note"] == "detector_missed"
    assert entry["detection_index"] is None
    assert entry["bbox"] is None
    assert entry["detector_conf"] is None


def test_recheck_confirms_blank():
    pred = {"filepath": "a.jpg", "detections": [],
            "whole_image_recheck": whole_image(BLANK, {"class": BLANK, "score": 0.99, "source": "classifier"})}
    [entry] = image_entries(pred, threshold=0.1)
    assert entry["category"] == "blank"
    assert "note" not in entry


def test_blank_without_recheck():
    [entry] = image_entries({"filepath": "a.jpg", "detections": []}, threshold=0.1)
    assert entry["category"] == "blank"
    assert entry["prediction_source"] == "detector_only"


def test_unreadable_image_is_an_error():
    pred = {"filepath": "bad.jpg", "failures": ["CLASSIFIER", "DETECTOR"]}
    [entry] = image_entries(pred, threshold=0.1)
    assert entry["category"] == "error"
    assert entry["failures"] == ["CLASSIFIER", "DETECTOR"]


def test_build_results_summary():
    predictions = [
        {"filepath": "a.jpg", "detections": [
            detection(raw=DEER, processed={"class": DEER, "score": 0.9, "source": "classifier"}),
            detection(raw=DEER, processed={"class": CERVIDAE, "score": 0.9, "source": "classifier+rollup_to_family"}),
            detection("human"),
        ]},
        {"filepath": "b.jpg", "detections": [],
         "whole_image_recheck": whole_image(DEER, {"class": ANIMAL, "score": 0.7, "source": "detector"})},
        {"filepath": "c.jpg", "detections": []},
        {"filepath": "d.jpg", "failures": ["DETECTOR"]},
    ]
    results = build_results(predictions, {"country": "USA"}, threshold=0.1)
    s = results["summary"]
    assert s["total_images"] == 4
    assert s["total_detections"] == 5  # the error entry isn't a detection
    assert (s["animals"], s["humans"], s["blanks"], s["errors"]) == (3, 1, 1, 1)
    assert s["detector_missed"] == 1
    assert s["confidence_rollups"] == 1
    assert results["metadata"]["geofencing_applied"] is True


# ---- species counts and CSV output ----

def test_species_counts_by_detection_and_image():
    predictions = [
        {"filepath": "/data/a.jpg", "detections": [
            detection(raw=DEER, processed={"class": DEER, "score": 0.9, "source": "classifier"}),
            detection(raw=DEER, processed={"class": DEER, "score": 0.9, "source": "classifier"}),
            detection("human"),
        ]},
        {"filepath": "/data/b.jpg", "detections": [
            detection(raw=DEER, processed={"class": DEER, "score": 0.9, "source": "classifier"}),
            detection(raw=DEER, processed={"class": CERVIDAE, "score": 0.9, "source": "classifier+rollup_to_family"}),
        ]},
        {"filepath": "/data/c.jpg", "detections": []},
    ]
    species = build_results(predictions, {}, threshold=0.1)["summary"]["species"]
    assert species == {
        "white-tailed deer": {"detections": 3, "images": 2, "events": 0},  # no timestamps
        "cervidae family": {"detections": 1, "images": 1, "events": 0},
    }
    assert list(species) == ["white-tailed deer", "cervidae family"]


def test_write_csvs(tmp_path):
    predictions = [
        {"filepath": str(tmp_path / "site1" / "a.jpg"), "detections": [
            detection(bbox=(0.123456, 0.2, 0.3, 0.4), raw=DEER,
                      processed={"class": DEER, "score": 0.987654, "source": "classifier"}),
        ]},
        {"filepath": str(tmp_path / "b.jpg"), "detections": []},
    ]
    results = build_results(predictions, {}, threshold=0.1)
    write_csv(results, str(tmp_path / "out" / "results.csv"), str(tmp_path))
    write_species_csv(results, str(tmp_path / "out" / "species_counts.csv"))

    with open(tmp_path / "out" / "results.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["file"] for r in rows] == [os.path.join("site1", "a.jpg"), "b.jpg"]
    assert rows[0]["prediction"] == "white-tailed deer"
    assert rows[0]["species_conf"] == "0.9877"
    assert rows[0]["bbox_x"] == "0.1235"
    assert rows[1]["category"] == "blank" and rows[1]["bbox_x"] == ""

    with open(tmp_path / "out" / "species_counts.csv", newline="") as f:
        assert list(csv.reader(f)) == [["species", "detections", "images", "events"],
                                       ["white-tailed deer", "1", "1", "0"]]
