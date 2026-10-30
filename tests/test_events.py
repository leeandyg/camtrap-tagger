from camtrap_tagger.events import build_events, fold_labels

DEER = "u1;mammalia;artiodactyla;cervidae;odocoileus;virginianus;white-tailed deer"
MULE = "u2;mammalia;artiodactyla;cervidae;odocoileus;hemionus;mule deer"
CERVIDAE = "u3;mammalia;artiodactyla;cervidae;;;cervidae family"
ODOCOILEUS = "u4;mammalia;artiodactyla;cervidae;odocoileus;;odocoileus species"
TURKEY = "u5;aves;galliformes;phasianidae;meleagris;gallopavo;wild turkey"
ANIMAL = "u6;;;;;;animal"


def entry(path, time, label=DEER, category="animal", deployment="camA", conf=0.9):
    name = label.split(";")[-1] if label else category
    return {"filepath": path, "deployment_id": deployment, "timestamp": time,
            "category": category, "prediction": name, "ensemble_prediction_raw": label,
            "species_conf": conf}


def blank(path, time, deployment="camA"):
    return entry(path, time, label=None, category="blank", deployment=deployment)


def observations(event):
    return {o["label"]: o["count"] for o in event["observations"]}


# ---- grouping ----

def test_images_within_gap_chain_into_one_event():
    entries = [
        entry("1.jpg", "2024-05-01T10:00:00"),
        entry("2.jpg", "2024-05-01T10:25:00"),
        entry("3.jpg", "2024-05-01T10:50:00"),  # 25 min after the previous image, 50 after the first
        entry("4.jpg", "2024-05-01T11:30:00"),
    ]
    events, event_of = build_events(entries, gap_minutes=30)
    assert [(e["event_id"], e["images"], e["start"], e["end"]) for e in events] == [
        ("camA-0001", 3, "2024-05-01T10:00:00", "2024-05-01T10:50:00"),
        ("camA-0002", 1, "2024-05-01T11:30:00", "2024-05-01T11:30:00"),
    ]
    assert event_of == {"1.jpg": "camA-0001", "2.jpg": "camA-0001", "3.jpg": "camA-0001",
                        "4.jpg": "camA-0002"}


def test_deployments_are_grouped_separately():
    entries = [entry("a.jpg", "2024-05-01T10:00:00", deployment="camA"),
               entry("b.jpg", "2024-05-01T10:01:00", deployment="camB")]
    events, _ = build_events(entries)
    assert sorted(e["event_id"] for e in events) == ["camA-0001", "camB-0001"]


def test_blank_and_untimed_images_are_left_out():
    entries = [
        entry("1.jpg", "2024-05-01T10:00:00"),
        blank("2.jpg", "2024-05-01T10:20:00"),  # doesn't extend the event
        entry("3.jpg", "2024-05-01T10:45:00"),
        entry("4.jpg", None),
    ]
    events, event_of = build_events(entries, gap_minutes=30)
    assert len(events) == 2
    assert "2.jpg" not in event_of and "4.jpg" not in event_of


# ---- counting ----

def test_count_is_max_in_one_image_not_sum_over_burst():
    entries = [
        entry("1.jpg", "2024-05-01T10:00:00"), entry("1.jpg", "2024-05-01T10:00:00"),
        entry("2.jpg", "2024-05-01T10:00:05"),
        entry("3.jpg", "2024-05-01T10:00:10"),
    ]
    [event] = build_events(entries)[0]
    assert observations(event) == {"white-tailed deer": 2}


def test_mixed_species_event_keeps_both():
    entries = [entry("1.jpg", "2024-05-01T10:00:00"),
               entry("1.jpg", "2024-05-01T10:00:00", label=TURKEY, conf=0.7),
               entry("2.jpg", "2024-05-01T10:00:05", category="human", label=None)]
    [event] = build_events(entries)[0]
    assert observations(event) == {"white-tailed deer": 1, "wild turkey": 1, "human": 1}
    turkey = next(o for o in event["observations"] if o["label"] == "wild turkey")
    assert turkey == {"label": "wild turkey", "category": "animal", "count": 1, "confidence": 0.7}


def test_broader_labels_fold_into_the_specific_one():
    entries = [
        entry("1.jpg", "2024-05-01T10:00:00", label=DEER),
        entry("2.jpg", "2024-05-01T10:00:05", label=CERVIDAE),
        entry("3.jpg", "2024-05-01T10:00:10", label=ANIMAL),
        entry("4.jpg", "2024-05-01T10:00:15", label=None, category="unknown"),
    ]
    [event] = build_events(entries)[0]
    assert observations(event) == {"white-tailed deer": 1}
    assert event["observations"][0]["category"] == "animal"


# ---- folding rules ----

def ranks(label):
    return (label.split(";")[1:6] + [""] * 5)[:5]


def test_fold_into_deepest_single_descendant():
    folded = fold_labels({"cervidae family": ranks(CERVIDAE), "odocoileus species": ranks(ODOCOILEUS),
                          "white-tailed deer": ranks(DEER)})
    assert folded == {"cervidae family": "white-tailed deer", "odocoileus species": "white-tailed deer",
                      "white-tailed deer": "white-tailed deer"}


def test_ambiguous_broad_label_stays():
    folded = fold_labels({"cervidae family": ranks(CERVIDAE), "white-tailed deer": ranks(DEER),
                          "mule deer": ranks(MULE), "animal": ranks(ANIMAL)})
    assert folded["cervidae family"] == "cervidae family"
    assert folded["animal"] == "animal"
    assert folded["mule deer"] == "mule deer"
