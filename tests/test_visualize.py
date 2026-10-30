from camtrap_tagger.visualize import (
    check_overlap,
    find_label_position,
    format_label,
    sort_detections_for_drawing,
)

IMG = (1000, 800)
TEXT_W, TEXT_H, PAD = 100, 20, 4
LABEL_W, LABEL_H = TEXT_W + 2 * PAD, TEXT_H + 2 * PAD


def place(bbox, placed=()):
    return find_label_position(bbox, TEXT_W, TEXT_H, IMG, list(placed), PAD)


# ---- label placement ----

def test_check_overlap():
    assert check_overlap((0, 0, 10, 10), (5, 5, 15, 15))
    assert not check_overlap((0, 0, 10, 10), (11, 0, 20, 10))


def test_label_goes_above_box_when_there_is_room():
    (x, y), rect = place((200, 300, 400, 500))
    assert (x, y) == (200, 300 - LABEL_H - 2)
    assert rect == (x, y, x + LABEL_W, y + LABEL_H)


def test_label_clamped_inside_image_at_top_edge():
    (_, y), _ = place((200, 0, 400, 200))
    assert y == 0


def test_label_clamped_inside_image_at_right_edge():
    (x, _), _ = place((950, 300, 1000, 400))
    assert x == IMG[0] - LABEL_W


def test_label_avoids_existing_label():
    _, first = place((200, 300, 400, 500))
    _, second = place((210, 300, 400, 500), placed=[first])
    assert not check_overlap(first, second)


def test_falls_back_to_first_spot_when_everything_overlaps():
    everything = (0, 0, *IMG)
    (x, y), _ = place((200, 300, 400, 500), placed=[everything])
    assert (x, y) == (200, 300 - LABEL_H - 2)


# ---- label text and draw order ----

def test_format_label_animal():
    lines = format_label({"category": "animal", "prediction": "red fox",
                          "detector_conf": 0.95, "species_conf": 0.874})
    assert lines == ["animal: 95%", "red fox: 87%"]


def test_format_label_detector_missed_animal():
    lines = format_label({"category": "animal", "prediction": "red fox",
                          "detector_conf": None, "species_conf": 0.8})
    assert lines == ["animal", "red fox: 80%"]


def test_format_label_human_uses_detector_conf():
    lines = format_label({"category": "human", "prediction": "human",
                          "detector_conf": 0.92, "species_conf": 0.5})
    assert lines == ["human: 92%"]


def test_format_label_unknown():
    assert format_label({"category": "unknown", "prediction": "unknown", "species_conf": 0.4}) == ["unknown", "40%"]


def test_draw_order_vehicles_then_humans_then_animals_by_confidence():
    dets = [
        {"category": "animal", "detector_conf": 0.9},
        {"category": "human", "detector_conf": 0.8},
        {"category": "animal", "detector_conf": 0.5},
        {"category": "vehicle", "detector_conf": 0.99},
    ]
    order = [(d["category"], d["detector_conf"]) for d in sort_detections_for_drawing(dets)]
    assert order == [("vehicle", 0.99), ("human", 0.8), ("animal", 0.5), ("animal", 0.9)]
