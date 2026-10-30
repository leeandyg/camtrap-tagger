import csv
import json
import os
from datetime import datetime, timedelta, timezone

import jsonschema
import pytest
from frictionless import Package
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT7

from camtrap_tagger.camtrapdp import (
    parse_contributor,
    parse_timezone,
    scientific_name,
    write_camtrap_dp,
)
from camtrap_tagger.cli import parse_args
from camtrap_tagger.metadata import Deployment
from camtrap_tagger.results import build_results

SCHEMAS = os.path.join(os.path.dirname(__file__), "data", "camtrap-dp-1.0.2")
DEER = "u1;mammalia;artiodactyla;cervidae;odocoileus;virginianus;white-tailed deer"
TURKEY = "u5;aves;galliformes;phasianidae;meleagris;gallopavo;wild turkey"
HUMAN = "990ae9dd-7a59-4344-afcb-1b7b21368000;mammalia;primates;hominidae;homo;sapiens;human"
LA = parse_timezone("America/Los_Angeles")


def detection(label, category="1", conf=0.9, bbox=(0.1, 0.2, 0.3, 0.4)):
    return {"category": category, "label": {"1": "animal", "2": "human"}[category], "conf": conf,
            "bbox": list(bbox), "raw_classifications": {"classes": [label], "scores": [0.9]},
            "processed_classification": {"class": label, "score": 0.88, "source": "classifier"}}


def make_results(root):
    """Two deployments in the sheet (one photo each burst), one uncovered folder, one untimed image."""
    def path(rel):
        return os.path.join(root, rel)

    predictions = [
        {"filepath": path("camA/1.jpg"), "detections": [detection(DEER), detection(TURKEY, bbox=(0.5, 0.5, 0.2, 0.2))]},
        {"filepath": path("camA/2.jpg"), "detections": [detection(DEER)]},
        {"filepath": path("camA/3.jpg"), "detections": []},
        {"filepath": path("camB/1.jpg"), "detections": [detection(HUMAN, category="2")]},
        {"filepath": path("camB/2.jpg"), "detections": [detection(DEER)]},
        {"filepath": path("camC/1.jpg"), "detections": [detection(DEER)]},
    ]
    info = {
        path("camA/1.jpg"): ("camA", "2024-07-01T10:00:00"),
        path("camA/2.jpg"): ("camA", "2024-07-01T10:00:05"),
        path("camA/3.jpg"): ("camA", "2024-07-02T08:00:00"),
        path("camB/1.jpg"): ("camB", "2024-12-01T09:00:00"),
        path("camB/2.jpg"): ("camB", None),
        path("camC/1.jpg"): ("camC", "2024-07-01T12:00:00"),
    }
    image_info = {p: {"deployment_id": d, "timestamp": t, "timestamp_source": "exif" if t else None,
                      "outside_deployment": False} for p, (d, t) in info.items()}
    deployments = [
        Deployment("camA", "camA", 44.1, -123.4, datetime(2024, 6, 1), datetime(2024, 8, 31, 23, 59, 59), "Ridge"),
        Deployment("camB", "camB", 44.2, -123.5, datetime(2024, 11, 1), datetime(2025, 1, 31, 23, 59, 59)),
        Deployment("camC", "camC"),  # folder not in the sheet: no coordinates or dates
    ]
    metadata = {"model": "kaggle:google/speciesnet/pyTorch/v4.0.3a/1", "threshold": 0.1}
    return build_results(predictions, metadata, 0.1, image_info, deployments)


@pytest.fixture
def package(tmp_path):
    results = make_results(str(tmp_path / "images"))
    out = tmp_path / "dp"
    warnings = write_camtrap_dp(results, str(out), str(tmp_path / "images"), LA, "Test project",
                                [parse_contributor("Ann Lee <ann@example.org>")], "targeted")
    return out, warnings


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---- validation against the official Camtrap DP 1.0.2 schemas ----

def validate_tables(out):
    descriptor = json.loads((out / "datapackage.json").read_text())
    # Use local copies of the official table schemas (no network in tests). Inlined schemas
    # break the Camtrap DP profile's "schema must be a versioned URL" rule, so this copy uses
    # the generic data-package profile; the Camtrap DP profile is checked in the next test.
    descriptor["profile"] = "data-package"
    for resource in descriptor["resources"]:
        with open(os.path.join(SCHEMAS, f"{resource['name']}-table-schema.json")) as f:
            resource["schema"] = json.load(f)
    (out / "local.json").write_text(json.dumps(descriptor))
    return Package(str(out / "local.json")).validate()


def test_tables_valid_against_camtrap_dp_schemas(package):
    out, _ = package
    report = validate_tables(out)
    assert report.valid, report.flatten(["rowNumber", "fieldName", "type", "note"])


def test_validation_catches_a_broken_table(package):
    out, _ = package
    path = out / "observations.csv"
    path.write_text(path.read_text().replace(",animal,", ",cat,", 1))
    report = validate_tables(out)
    assert not report.valid
    assert ["observationType", "constraint-error"] in report.flatten(["fieldName", "type"])


def test_descriptor_valid_against_camtrap_dp_profile(package):
    out, _ = package
    descriptor = json.loads((out / "datapackage.json").read_text())
    with open(os.path.join(SCHEMAS, "camtrap-dp-profile.json")) as f:
        profile = json.load(f)
    with open(os.path.join(SCHEMAS, "geojson.json")) as f:
        geojson = Resource.from_contents(json.load(f), default_specification=DRAFT7)
    # The profile's first part is the generic Frictionless schema (checked by frictionless itself);
    # the second part holds the Camtrap DP rules.
    schema = {**profile["allOf"][1], "$defs": profile.get("$defs", {})}
    registry = Registry().with_resource("http://json.schemastore.org/geojson.json", geojson)
    validator = jsonschema.Draft7Validator(schema, registry=registry,
                                           format_checker=jsonschema.FormatChecker())
    assert [e.message for e in validator.iter_errors(descriptor)] == []


# ---- content ----

def test_left_out_data_is_reported(package):
    out, warnings = package
    assert any("camC" in w for w in warnings)
    assert any("1 file(s) without a capture time" in w for w in warnings)
    assert [d["deploymentID"] for d in read(out / "deployments.csv")] == ["camA", "camB"]
    assert len(read(out / "media.csv")) == 4


def test_timestamps_get_the_offset_in_force_on_that_date(package):
    out, _ = package
    deployments = {d["deploymentID"]: d for d in read(out / "deployments.csv")}
    assert deployments["camA"]["deploymentStart"] == "2024-06-01T00:00:00-07:00"  # daylight time
    assert deployments["camB"]["deploymentStart"] == "2024-11-01T00:00:00-07:00"
    assert deployments["camB"]["deploymentEnd"] == "2025-01-31T23:59:59-08:00"   # standard time


def test_media_and_event_observations(package):
    out, _ = package
    rows = read(out / "observations.csv")
    media_rows = [r for r in rows if r["observationLevel"] == "media"]
    event_rows = [r for r in rows if r["observationLevel"] == "event"]

    assert sorted(r["observationType"] for r in media_rows) == ["animal", "animal", "animal", "blank", "human"]
    turkey = next(r for r in media_rows if r["scientificName"] == "Meleagris gallopavo")
    assert (turkey["bboxX"], turkey["bboxWidth"], turkey["count"]) == ("0.5", "0.2", "1")
    assert turkey["classificationMethod"] == "machine"
    assert turkey["classifiedBy"].startswith("SpeciesNet v4.0.3a")

    # The two camA photos 5 s apart are one event with one deer and one turkey.
    camA = sorted((r["scientificName"], r["count"]) for r in event_rows if r["deploymentID"] == "camA")
    assert camA == [("Meleagris gallopavo", "1"), ("Odocoileus virginianus", "1")]
    assert all(r["mediaID"] == "" for r in event_rows)


def test_package_metadata(package):
    out, _ = package
    d = json.loads((out / "datapackage.json").read_text())
    assert d["project"]["samplingDesign"] == "targeted"
    assert d["contributors"] == [{"title": "Ann Lee", "email": "ann@example.org", "role": "contact"}]
    assert d["temporal"] == {"start": "2024-06-01", "end": "2025-01-31"}
    assert d["spatial"]["type"] == "Polygon"
    assert [t["scientificName"] for t in d["taxonomic"]] == ["Meleagris gallopavo", "Odocoileus virginianus"]
    assert d["taxonomic"][1]["vernacularNames"] == {"eng": "white-tailed deer"}


# ---- helpers and CLI checks ----

def test_parse_timezone():
    assert parse_timezone("-07:00").utcoffset(None) == timedelta(hours=-7)
    assert parse_timezone("+0530").utcoffset(None) == timedelta(hours=5, minutes=30)
    assert parse_timezone("UTC").utcoffset(datetime(2024, 1, 1)) == timedelta(0)
    with pytest.raises(ValueError, match="unknown time zone"):
        parse_timezone("Mars/Olympus")


def test_scientific_name():
    assert scientific_name(["mammalia", "artiodactyla", "cervidae", "odocoileus", "virginianus"]) \
        == ("Odocoileus virginianus", "species")
    assert scientific_name(["mammalia", "artiodactyla", "cervidae", "", ""]) == ("Cervidae", "family")
    assert scientific_name(["", "", "", "", ""]) == (None, None)


def test_parse_contributor():
    assert parse_contributor("Ann Lee") == {"title": "Ann Lee", "role": "contact"}


@pytest.mark.parametrize("argv, message", [
    (["--camtrap-dp", "--deployments", "s.csv"], "needs --timezone"),
    (["--camtrap-dp", "--timezone", "UTC"], "needs --deployments"),
    (["--timezone", "Mars/Olympus"], "unknown time zone"),
])
def test_cli_checks_camtrap_dp_options(argv, message, capsys):
    with pytest.raises(SystemExit):
        parse_args(["photos", *argv])
    assert message in capsys.readouterr().err


def test_utc_offset_from_name_matches_timezone_module():
    assert datetime(2024, 7, 1, tzinfo=LA).utcoffset() == timezone(timedelta(hours=-7)).utcoffset(None)
