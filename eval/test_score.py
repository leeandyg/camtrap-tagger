from score import grade, is_ancestor, iou, match, score_image, score_image_level

COTTONTAIL = "667a4650-a141-4c4e-844e-58cdeaeb4ae1"
CROW = "87fdd451-ca1e-47b1-af43-aa4517cd13bf"
BIRD = "b1352069-a39c-4a84-a949-60044271c0c1"
TAXONOMY = {
    COTTONTAIL: ["mammalia", "lagomorpha", "leporidae", "sylvilagus", "floridanus"],
    CROW: ["aves", "passeriformes", "corvidae", "corvus", "brachyrhynchos"],
    BIRD: ["aves", "", "", "", ""],
}
COTTONTAIL_LABEL = f"{COTTONTAIL};mammalia;lagomorpha;leporidae;sylvilagus;floridanus;eastern cottontail"
LEPORIDAE_LABEL = "x;mammalia;lagomorpha;leporidae;;;rabbit and hare family"
CROW_LABEL = f"{CROW};aves;passeriformes;corvidae;corvus;brachyrhynchos;american crow"
ANIMAL_LABEL = "1f689929-883d-4dae-958c-3d57ab5b6c16;;;;;;animal"


def entry(label, category="animal", prediction="x", bbox=(0.1, 0.1, 0.2, 0.2), **extra):
    return {"category": category, "prediction": prediction, "ensemble_prediction_raw": label,
            "bbox": list(bbox) if bbox else None, **extra}


def test_is_ancestor():
    fine = TAXONOMY[COTTONTAIL]
    assert is_ancestor(["mammalia", "lagomorpha", "leporidae", "", ""], fine)
    assert not is_ancestor(fine, fine)
    assert not is_ancestor(["aves", "", "", "", ""], fine)
    assert not is_ancestor(["", "", "", "", ""], fine)


def test_grades():
    assert grade(entry(COTTONTAIL_LABEL), COTTONTAIL, TAXONOMY) == "exact"
    assert grade(entry(LEPORIDAE_LABEL), COTTONTAIL, TAXONOMY) == "coarser"
    assert grade(entry(ANIMAL_LABEL, prediction="animal"), COTTONTAIL, TAXONOMY) == "animal"
    assert grade(entry(CROW_LABEL), BIRD, TAXONOMY) == "finer"
    assert grade(entry(CROW_LABEL), COTTONTAIL, TAXONOMY) == "wrong"
    assert grade(entry(None, category="human"), COTTONTAIL, TAXONOMY) == "wrong"
    assert grade(entry(None, category="unknown"), COTTONTAIL, TAXONOMY) == "unknown"


def test_domestic_form_counts_as_exact():
    junglefowl = "89851bea-765b-431a-bb03-6f821931ebdc"
    taxonomy = {junglefowl: ["aves", "galliformes", "phasianidae", "gallus", "gallus"]}
    chicken = "cae9534f;aves;galliformes;phasianidae;gallus;gallus domesticus;domestic chicken"
    grey = "5d6912b0;aves;galliformes;phasianidae;gallus;sonneratii;grey junglefowl"
    assert grade(entry(chicken), junglefowl, taxonomy) == "exact"
    assert grade(entry(grey), junglefowl, taxonomy) == "wrong"


def test_iou_and_matching():
    assert iou([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
    assert iou([0, 0, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]) == 0.0
    truths = [{"bbox": [0, 0, 0.2, 0.2]}, {"bbox": [0.5, 0.5, 0.2, 0.2]}]
    preds = [{"bbox": [0.51, 0.5, 0.2, 0.2]}, {"bbox": [0.9, 0.9, 0.1, 0.1]}]
    assert match(truths, preds) == {1: 0}


def test_score_image_counts_extra_and_missed():
    truth = {"objects": [
        {"bbox": [0.1, 0.1, 0.2, 0.2], "wi_guid": COTTONTAIL, "category": "animal"},
        {"bbox": [0.6, 0.6, 0.2, 0.2], "wi_guid": CROW, "category": "animal"},
    ]}
    entries = [entry(COTTONTAIL_LABEL), entry(None, category="human", bbox=(0.4, 0.0, 0.1, 0.1))]
    s = score_image(truth, entries, TAXONOMY)
    assert s["animals"] == {"exact": 1, "missed": 1}
    assert s["extra"] == {"human": 1}
    assert not s["all_animals_correct"]


def test_score_image_recheck_recovers_one_animal():
    truth = {"objects": [{"bbox": [0.1, 0.1, 0.2, 0.2], "wi_guid": CROW, "category": "animal"}]}
    entries = [entry(CROW_LABEL, bbox=None, note="detector_missed")]
    assert score_image(truth, entries, TAXONOMY)["animals"] == {"recovered": 1}


def test_score_image_level_takes_best_label_and_counts_extras():
    truth = {"objects": [{"bbox": None, "wi_guid": COTTONTAIL, "category": "animal"}]}
    entries = [entry(LEPORIDAE_LABEL), entry(COTTONTAIL_LABEL), entry(CROW_LABEL)]
    s = score_image_level(truth, entries, TAXONOMY)
    assert s["animals"] == {"exact": 1}
    assert s["extra"] == {"animal": 2}
    assert s["all_animals_correct"]


def test_score_image_level_recheck_and_missed():
    truth = {"objects": [{"bbox": None, "wi_guid": CROW, "category": "animal"}]}
    recheck = [entry(CROW_LABEL, bbox=None, note="detector_missed")]
    assert score_image_level(truth, recheck, TAXONOMY)["animals"] == {"recovered": 1}
    blank = [entry(None, category="blank", bbox=None)]
    assert score_image_level(truth, blank, TAXONOMY)["animals"] == {"missed": 1}


def test_score_image_empty():
    s = score_image({"objects": []}, [entry(None, category="blank", bbox=None)], TAXONOMY)
    assert s["empty_correct"]
