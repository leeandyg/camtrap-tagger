"""Capture times and deployments (one camera at one place for one period).

Timestamps come from EXIF (DateTimeOriginal, else DateTime), optionally falling back to the
file's modification time. They are local camera time without a time zone, as cameras record them.

By default each folder of images is its own deployment. A deployment sheet (CSV) can map
folders to deployment IDs and add coordinates and start/end times:

    deploymentID,folder,latitude,longitude,start,end,locationName
    cam-A-2024,site1/camA,44.12,-123.45,2024-05-01,2024-08-31,North ridge
"""

import csv
import difflib
import os
from dataclasses import dataclass
from datetime import datetime

from PIL import Image

EXIF_IFD = 0x8769
DATETIME_ORIGINAL = 36867
DATETIME = 306

REQUIRED_COLUMNS = ["deploymentID", "folder"]
OPTIONAL_COLUMNS = ["latitude", "longitude", "start", "end", "locationName"]


@dataclass
class Deployment:
    deployment_id: str
    folder: str  # relative to the input folder; "." is the input folder itself
    latitude: float | None = None
    longitude: float | None = None
    start: datetime | None = None
    end: datetime | None = None
    location_name: str | None = None

    def contains(self, timestamp):
        """False if timestamp falls outside this deployment's start/end (when both are known)."""
        if timestamp is None or self.start is None or self.end is None:
            return True
        return self.start <= datetime.fromisoformat(timestamp) <= self.end


class DeploymentSheetError(ValueError):
    """A deployment sheet that can't be used; .problems lists every issue found."""

    def __init__(self, problems):
        self.problems = problems
        super().__init__("\n".join(problems))


def read_timestamp(path, file_time_fallback=False):
    """(ISO 8601 local time, source) for an image, or (None, None) if unknown.

    Source is "exif", or "file_modified" when file_time_fallback is set. A file's modified
    time is only the capture time if the file was copied off the card with dates preserved;
    downloaded or re-saved files get the download time, so it is off by default.
    """
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            raw = exif.get_ifd(EXIF_IFD).get(DATETIME_ORIGINAL) or exif.get(DATETIME)
        if raw:
            parsed = datetime.strptime(str(raw).strip("\x00 "), "%Y:%m:%d %H:%M:%S")
            return parsed.isoformat(), "exif"
    except (OSError, ValueError):
        pass
    if file_time_fallback:
        try:
            modified = datetime.fromtimestamp(os.path.getmtime(path)).replace(microsecond=0)
            return modified.isoformat(), "file_modified"
        except OSError:
            pass
    return None, None


def normalize_folder(folder):
    return os.path.normpath(folder.strip().strip("/")) if folder.strip() else "."


def image_folder(filepath, source_root):
    return os.path.dirname(os.path.relpath(filepath, source_root)) or "."


def _parse_time(value, end_of_day):
    """Parse a sheet date or datetime; a bare date as an end time means the end of that day."""
    parsed = datetime.fromisoformat(value.strip())
    if end_of_day and len(value.strip()) == 10:
        parsed = parsed.replace(hour=23, minute=59, second=59)
    return parsed


def load_deployment_sheet(path, source_root):
    """Read and validate a deployment sheet. Raises DeploymentSheetError listing all problems."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        rows = list(reader)

    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        expected = ", ".join(REQUIRED_COLUMNS + OPTIONAL_COLUMNS)
        raise DeploymentSheetError([(f"Missing required column(s): {', '.join(missing)} "
                                     f"(expected {expected})")])

    existing = sorted({
        os.path.relpath(root, source_root)
        for root, _, files in os.walk(source_root) if files
    })
    problems, deployments, seen_ids, seen_folders = [], [], set(), set()
    for line, row in enumerate(rows, start=2):  # line 1 is the header
        def get(col, row=row):
            return (row.get(col) or "").strip()

        dep_id, folder = get("deploymentID"), get("folder")
        if not dep_id or not folder:
            problems.append(f"Line {line}: deploymentID and folder are required")
            continue
        folder = normalize_folder(folder)
        if dep_id in seen_ids:
            problems.append(f"Line {line}: duplicate deploymentID '{dep_id}'")
        if folder in seen_folders:
            problems.append(f"Line {line}: folder '{folder}' is listed twice")
        seen_ids.add(dep_id)
        seen_folders.add(folder)

        if not os.path.isdir(os.path.join(source_root, folder)):
            hint = difflib.get_close_matches(folder, existing, n=1)
            problems.append(f"Line {line}: folder '{folder}' not found in the input folder"
                            + (f" (did you mean '{hint[0]}'?)" if hint else ""))

        dep = Deployment(dep_id, folder, location_name=get("locationName") or None)
        lat, lon = get("latitude"), get("longitude")
        if lat or lon:
            try:
                dep.latitude, dep.longitude = float(lat), float(lon)
                if not (-90 <= dep.latitude <= 90 and -180 <= dep.longitude <= 180):
                    problems.append(f"Line {line}: coordinates {lat}, {lon} are out of range")
            except ValueError:
                problems.append(f"Line {line}: latitude and longitude must both be numbers "
                                f"(got '{lat}', '{lon}')")
        for col, end_of_day in (("start", False), ("end", True)):
            if get(col):
                try:
                    setattr(dep, col, _parse_time(get(col), end_of_day))
                except ValueError:
                    problems.append(f"Line {line}: can't read {col} '{get(col)}' "
                                    "(use YYYY-MM-DD or YYYY-MM-DD HH:MM)")
        if dep.start and dep.end and dep.start >= dep.end:
            problems.append(f"Line {line}: start is not before end")
        deployments.append(dep)

    if not rows:
        problems.append("The sheet has no deployments")
    if problems:
        raise DeploymentSheetError(problems)
    return deployments


def assign_deployments(filepaths, source_root, sheet=None):
    """Map each image to a deployment.

    Images go to the sheet deployment whose folder contains them (deepest match wins);
    images not covered by the sheet get one deployment per folder, named after the folder.
    Returns (deployment ID per filepath, all deployments, warnings).
    """
    sheet = sheet or []
    by_folder = sorted(sheet, key=lambda d: d.folder.count(os.sep) + (d.folder != "."), reverse=True)
    deployments = {d.deployment_id: d for d in sheet}
    assigned, uncovered, used = {}, set(), set()

    for path in filepaths:
        folder = image_folder(path, source_root)
        match = next((d for d in by_folder
                      if d.folder == "." or folder == d.folder
                      or folder.startswith(d.folder + os.sep)), None)
        if match is None:
            dep_id = folder if folder != "." else os.path.basename(os.path.abspath(source_root))
            match = deployments.setdefault(dep_id, Deployment(dep_id, folder))
            if sheet:
                uncovered.add(folder)
        assigned[path] = match.deployment_id
        used.add(match.deployment_id)

    warnings = []
    if uncovered:
        warnings.append(f"{len(uncovered)} folder(s) with images are not in the deployment sheet "
                        f"and got their own deployment: {', '.join(sorted(uncovered))}")
    unused = [d.deployment_id for d in sheet if d.deployment_id not in used]
    if unused:
        warnings.append(f"Deployment(s) with no images: {', '.join(unused)}")
    return assigned, list(deployments.values()), warnings
