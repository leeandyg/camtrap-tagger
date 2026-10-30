import csv

from camtrap_tagger.results import build_results, write_csv
from camtrap_tagger.review import review_reasons, write_review_csv

DEER = "u1;mammalia;artiodactyla;cervidae;odocoileus;virginianus;white-tailed deer"
CERVIDAE = "u2;mammalia;artiodactyla;cervidae;;;cervidae family"
ANIMAL = "u3;;;;;;animal"
HUMAN = "990ae9dd-7a59-4344-afcb-1b7b21368000;mammalia;primates;hominidae;homo;sapiens;human"
UNKNOWN = "f2efdae9-efb8-48fb-8a91-eccf79ab4ffb;no cv result;no cv result;no cv result;no cv result;no cv result;no cv result"
BLANK = "f1856211-cfb7-4a5b-9158-c0f72fd09ee6;;;;;;blank"


def detection(label, score, detector="animal", source="classifier"):
    category = {"animal": "1", "human": "2", "vehicle": "3"}[detector]
    return {"category": category, "label": detector, "conf": 0.9, "bbox": [0.1, 0.2, 0.3, 0.4],
            "raw_classifications": {"classes": [label], "scores": [score]},
            "processed_classification": {"class": label, "score": score, "source": source}}


def results_for(tmp_path, images, review_below=0.8):
    predictions = [{"filepath": str(tmp_path / name), "detections": dets} for name, dets in images]
    return build_results(predictions, {"model": "m"}, 0.1, review_below=review_below)


def reasons(tmp_path, det, review_below=0.8):
    [entry] = results_for(tmp_path, [("a.jpg", [det])], review_below)["predictions"]
    return entry["review_reasons"]


def test_confident_species_is_not_flagged(tmp_path):
    assert reasons(tmp_path, detection(DEER, 0.95)) == []


def test_low_confidence_uses_the_threshold(tmp_path):
    assert reasons(tmp_path, detection(DEER, 0.7)) == ["low_confidence"]
    assert reasons(tmp_path, detection(DEER, 0.7), review_below=0.6) == []


def test_labels_above_species_are_flagged(tmp_path):
    assert reasons(tmp_path, detection(CERVIDAE, 0.9, source="classifier+rollup_to_family")) == ["not_species_level"]
    assert reasons(tmp_path, detection(ANIMAL, 0.6, source="detector")) == ["not_species_level", "low_confidence"]


def test_geofence_rollups_are_outside_range(tmp_path):
    det = detection(CERVIDAE, 0.99, source="classifier+geofence+rollup_to_family")
    det["raw_classifications"]["classes"] = ["u4;mammalia;artiodactyla;cervidae;cervus;elaphus;red deer"]
    results = results_for(tmp_path, [("elk.jpg", [det])])
    assert results["predictions"][0]["review_reasons"] == ["outside_range"]
    write_review_csv(results, str(tmp_path / "review.csv"), str(tmp_path))
    [row] = list(csv.DictReader((tmp_path / "review.csv").open()))
    assert row["detections"] == "0: cervidae family 0.99 (top guess: red deer) [outside_range]"


def test_unknown_and_detector_disagreement(tmp_path):
    assert reasons(tmp_path, detection(UNKNOWN, 0.3)) == ["unknown"]
    assert reasons(tmp_path, detection(HUMAN, 0.99, detector="vehicle")) == ["detector_disagrees"]
    assert reasons(tmp_path, detection(HUMAN, 0.99, detector="human")) == []


def test_blanks_and_recheck(tmp_path):
    assert reasons(tmp_path, detection(BLANK, 0.5)) == []
    recheck = {"category": "0", "label": "whole_image_classification", "conf": 1.0, "bbox": [0, 0, 1, 1],
               "raw_classifications": {"classes": [DEER], "scores": [0.9]},
               "processed_classification": {"class": DEER, "score": 0.9, "source": "classifier"}}
    results = build_results([{"filepath": str(tmp_path / "a.jpg"), "detections": [],
                              "whole_image_recheck": recheck}], {"model": "m"}, 0.1)
    assert results["predictions"][0]["review_reasons"] == ["recheck_only"]


def test_unclassified_detections_are_not_flagged():
    # The crop backend leaves humans and vehicles unclassified.
    entry = {"category": "human", "detector_label": "human", "species_conf": None}
    assert review_reasons(entry) == []


def test_review_csv_lists_flagged_images_with_all_their_detections(tmp_path):
    results = results_for(tmp_path, [
        ("ok.jpg", [detection(DEER, 0.95)]),
        ("mixed.jpg", [detection(DEER, 0.95), detection(CERVIDAE, 0.9, source="classifier+rollup_to_family"),
                       detection(DEER, 0.7)]),
        ("empty.jpg", []),
    ])
    assert results["summary"]["review"] == {
        "images": 1, "detections": 2, "by_reason": {"not_species_level": 1, "low_confidence": 1}}

    out = tmp_path / "out" / "review.csv"
    assert write_review_csv(results, str(out), str(tmp_path)) == 1
    [row] = list(csv.DictReader(out.open()))
    assert row["file"] == "mixed.jpg"
    assert row["reasons"] == "not_species_level;low_confidence"
    assert row["lowest_conf"] == "0.7"
    assert row["detections"] == ("0: white-tailed deer 0.95; 1: cervidae family 0.90 [not_species_level]; "
                                 "2: white-tailed deer 0.70 [low_confidence]")


def test_review_reasons_column_in_results_csv(tmp_path):
    results = results_for(tmp_path, [("a.jpg", [detection(DEER, 0.7)])])
    write_csv(results, str(tmp_path / "results.csv"), str(tmp_path))
    [row] = list(csv.DictReader((tmp_path / "results.csv").open()))
    assert row["review_reasons"] == "low_confidence"
