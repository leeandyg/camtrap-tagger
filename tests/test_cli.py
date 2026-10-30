import pytest

from camtrap_tagger.cli import parse_args
from camtrap_tagger.pipeline import list_images


def test_defaults():
    args = parse_args(["photos"])
    assert args.backend == "upstream"
    assert args.threshold == 0.1
    assert args.country is None
    assert args.recheck is False


def test_location_is_uppercased():
    args = parse_args(["photos", "--country", "usa", "--admin1-region", "ca"])
    assert (args.country, args.admin1_region) == ("USA", "CA")


@pytest.mark.parametrize("argv", [
    ["--country", "US"],
    ["--country", "U5A"],
    ["--admin1-region", "CA"],
    ["--country", "KEN", "--admin1-region", "CA"],
    ["--country", "USA", "--admin1-region", "XX"],
    ["--threshold", "1.5"],
    ["--backend", "megadetector"],
])
def test_bad_arguments_are_rejected(argv, capsys):
    with pytest.raises(SystemExit):
        parse_args(["photos", *argv])
    assert "error" in capsys.readouterr().err


def test_list_images_is_recursive_and_skips_non_images(tmp_path):
    for name in ["a.jpg", "b.WEBP", "c.png", "notes.txt", ".hidden.jpg", "sub/d.jpeg", "sub/e.tif"]:
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.touch()
    found = [str(p.relative_to(tmp_path)) for p in map(type(tmp_path), list_images(tmp_path))]
    assert found == ["a.jpg", "b.WEBP", "c.png", "sub/d.jpeg", "sub/e.tif"]
