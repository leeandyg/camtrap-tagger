import json
import os

from camtrap_tagger.results import build_results
from camtrap_tagger.timelapse import build_recognition_file, write_recognition_file

DEER = "u1;mammalia;artiodactyla;cervidae;odocoileus;virginianus;white-tailed deer"
HUMAN = "990ae9dd-7a59-4344-afcb-1b7b21368000;mammalia;primates;hominidae;homo;sapiens;human"


def detection(label, detector="animal", conf=0.9, score=0.8, bbox=(0.1, 0.2, 0.3, 0.4)):
    category = {"animal": "1", "human": "2", "vehicle": "3"}[detector]
    return {"category": category, "label": detector, "conf": conf, "bbox": list(bbox),
            "raw_classifications": {"classes": [label], "scores": [score]},
            "processed_classification": {"class": label, "score": score, "source": "classifier"}}


def recognition(tmp_path, predictions):
    results = build_results(predictions, {"model": "kaggle:google/speciesnet/pyTorch/v4.0.3a/1",
                                          "threshold": 0.1}, 0.1)
    return build_recognition_file(results, str(tmp_path))


def test_boxes_keep_detector_category_and_carry_the_final_label(tmp_path):
    d = recognition(tmp_path, [{"filepath": str(tmp_path / "site1" / "a.jpg"), "detections": [
        detection(DEER),
        detection(HUMAN, detector="vehicle", conf=0.5, score=0.99),  # SpeciesNet overrules the detector
    ]}])
    [image] = d["images"]
    assert image["file"] == "site1/a.jpg"
    names = d["classification_categories"]
    deer, person = image["detections"]
    assert (deer["category"], deer["conf"], deer["bbox"]) == ("1", 0.9, [0.1, 0.2, 0.3, 0.4])
    assert [(names[c], p) for c, p in deer["classifications"]] == [("white-tailed deer", 0.8)]
    assert d["detection_categories"][person["category"]] == "vehicle"
    assert [names[c] for c, _ in person["classifications"]] == ["human"]
    deer_id = next(k for k, v in names.items() if v == "white-tailed deer")
    assert d["classification_category_descriptions"][deer_id] == "mammalia;artiodactyla;cervidae;odocoileus;virginianus"


def test_empty_and_unreadable_images(tmp_path):
    d = recognition(tmp_path, [
        {"filepath": str(tmp_path / "empty.jpg"), "detections": []},
        {"filepath": str(tmp_path / "bad.jpg"), "failures": ["CLASSIFIER", "DETECTOR"]},
    ])
    images = {i["file"]: i for i in d["images"]}
    assert images["empty.jpg"]["detections"] == []
    assert images["bad.jpg"]["detections"] is None and images["bad.jpg"]["failure"]


def test_whole_image_recheck_becomes_a_full_frame_box(tmp_path):
    recheck = {"category": "0", "label": "whole_image_classification", "conf": 1.0, "bbox": [0, 0, 1, 1],
               "raw_classifications": {"classes": [DEER], "scores": [0.9]},
               "processed_classification": {"class": DEER, "score": 0.9, "source": "classifier"}}
    d = recognition(tmp_path, [{"filepath": str(tmp_path / "a.jpg"), "detections": [],
                                "whole_image_recheck": recheck}])
    [box] = d["images"][0]["detections"]
    assert box["bbox"] == [0.0, 0.0, 1.0, 1.0] and box["category"] == "1"


def test_info_and_file_written(tmp_path):
    predictions = [{"filepath": str(tmp_path / "a.jpg"), "detections": [detection(DEER)]}]
    results = build_results(predictions, {"model": "m", "threshold": 0.1}, 0.1)
    out = tmp_path / "out" / "timelapse_recognition.json"
    write_recognition_file(results, str(out), str(tmp_path))
    d = json.loads(out.read_text())
    assert d["info"]["format_version"] == "1.6"
    assert d["detection_categories"] == {"1": "animal", "2": "person", "3": "vehicle"}
    assert os.path.sep not in d["images"][0]["file"] or os.path.sep == "/"
