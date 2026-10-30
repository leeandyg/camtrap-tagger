import os
from datetime import datetime

import pytest
from PIL import Image

from camtrap_tagger.metadata import (
    DATETIME_ORIGINAL,
    EXIF_IFD,
    Deployment,
    DeploymentSheetError,
    assign_deployments,
    load_deployment_sheet,
    read_timestamp,
)


def make_image(path, taken=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exif = Image.Exif()
    if taken:
        exif.get_ifd(EXIF_IFD)[DATETIME_ORIGINAL] = taken
    Image.new("RGB", (8, 8)).save(path, exif=exif)
    return str(path)


def write_sheet(path, text):
    path.write_text(text)
    return str(path)


# ---- timestamps ----

def test_timestamp_from_exif(tmp_path):
    path = make_image(tmp_path / "a.jpg", "2021:05:15 01:19:50")
    assert read_timestamp(path) == ("2021-05-15T01:19:50", "exif")


def test_no_exif_time_is_unknown_unless_file_times_requested(tmp_path):
    path = make_image(tmp_path / "a.jpg")
    os.utime(path, (datetime(2020, 1, 2, 3, 4, 5).timestamp(),) * 2)
    assert read_timestamp(path) == (None, None)
    assert read_timestamp(path, file_time_fallback=True) == ("2020-01-02T03:04:05", "file_modified")


def test_unparseable_exif_is_unknown(tmp_path):
    path = make_image(tmp_path / "a.jpg", "0000:00:00 00:00:00")
    assert read_timestamp(path) == (None, None)


# ---- deployment sheet ----

def test_load_sheet(tmp_path):
    os.makedirs(tmp_path / "site1" / "camA")
    sheet = write_sheet(tmp_path / "sheet.csv",
                        "deploymentID,folder,latitude,longitude,start,end,locationName\n"
                        "A,site1/camA/,44.5,-123.25,2024-05-01,2024-08-31,Ridge\n")
    [dep] = load_deployment_sheet(sheet, str(tmp_path))
    assert dep.folder == os.path.join("site1", "camA")
    assert (dep.latitude, dep.longitude, dep.location_name) == (44.5, -123.25, "Ridge")
    assert dep.start == datetime(2024, 5, 1)
    assert dep.end == datetime(2024, 8, 31, 23, 59, 59)  # a bare end date covers the whole day


def test_sheet_problems_are_all_reported(tmp_path):
    make_image(tmp_path / "camA" / "1.jpg")
    sheet = write_sheet(tmp_path / "sheet.csv",
                        "deploymentID,folder,latitude,longitude,start,end\n"
                        "A,camA,95,10,2024-05-01,2024-04-01\n"
                        "A,camB,,,,\n"
                        "C,camA,abc,1,nope,\n")
    with pytest.raises(DeploymentSheetError) as err:
        load_deployment_sheet(sheet, str(tmp_path))
    text = str(err.value)
    assert "out of range" in text
    assert "start is not before end" in text
    assert "duplicate deploymentID 'A'" in text
    assert "folder 'camB' not found" in text and "did you mean 'camA'" in text
    assert "listed twice" in text
    assert "must both be numbers" in text
    assert "can't read start 'nope'" in text


def test_sheet_missing_columns(tmp_path):
    sheet = write_sheet(tmp_path / "sheet.csv", "id,path\nA,camA\n")
    with pytest.raises(DeploymentSheetError, match="Missing required column"):
        load_deployment_sheet(sheet, str(tmp_path))


# ---- assigning images to deployments ----

def test_default_deployment_per_folder(tmp_path):
    paths = [make_image(tmp_path / "camA" / "1.jpg"), make_image(tmp_path / "2.jpg")]
    assigned, deployments, warnings = assign_deployments(paths, str(tmp_path))
    assert assigned == {paths[0]: "camA", paths[1]: os.path.basename(tmp_path)}
    assert len(deployments) == 2 and warnings == []


def test_sheet_deepest_folder_wins_and_uncovered_folders_warn(tmp_path):
    paths = [
        make_image(tmp_path / "site1" / "camA" / "1.jpg"),
        make_image(tmp_path / "site1" / "other" / "2.jpg"),
        make_image(tmp_path / "site2" / "3.jpg"),
    ]
    sheet = [Deployment("site1-all", "site1"),
             Deployment("A", os.path.join("site1", "camA")),
             Deployment("unused", "site3")]
    assigned, deployments, warnings = assign_deployments(paths, str(tmp_path), sheet)
    assert assigned == {paths[0]: "A", paths[1]: "site1-all", paths[2]: "site2"}
    assert {d.deployment_id for d in deployments} == {"site1-all", "A", "unused", "site2"}
    assert any("not in the deployment sheet" in w and "site2" in w for w in warnings)
    assert any("no images: unused" in w for w in warnings)


def test_deployment_window():
    dep = Deployment("A", "camA", start=datetime(2024, 5, 1), end=datetime(2024, 5, 31, 23, 59, 59))
    assert dep.contains("2024-05-31T12:00:00")
    assert not dep.contains("2024-06-01T00:00:00")
    assert dep.contains(None)
    assert Deployment("B", "camB").contains("1999-01-01T00:00:00")
