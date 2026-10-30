"""Export results as a Camtrap DP 1.0.2 data package.

Camtrap DP (https://camtrap-dp.tdwg.org) is the TDWG standard for camera-trap data, read by
GBIF and by R/Python tools such as camtrapdp, camtrapR and frictionless. A package is a folder
with datapackage.json and three tables:

- deployments.csv: one row per camera deployment (needs coordinates and start/end)
- media.csv: one row per image or video with a capture time
- observations.csv: one row per detection (observationLevel "media", with its bounding box;
  for a video, the times the label was on screen and no box, since a box belongs to one frame)
  and one row per species per event (observationLevel "event", with its count)

All observations are machine classifications. Camera times have no time zone, so the caller
supplies one; media `filePath` is relative to the original image folder.
"""

import csv
import hashlib
import json
import mimetypes
import os
import re
import uuid
from datetime import UTC, datetime, timedelta, timezone
from importlib.metadata import version
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from camtrap_tagger.events import label_ranks
from camtrap_tagger.metadata import VIDEO_MEDIATYPES

CAMTRAP_DP_VERSION = "1.0.2"
SCHEMA_BASE = f"https://raw.githubusercontent.com/tdwg/camtrap-dp/{CAMTRAP_DP_VERSION}"
SAMPLING_DESIGNS = ["simpleRandom", "systematicRandom", "clusteredRandom", "experimental",
                    "targeted", "opportunistic"]
REPOSITORY = "https://github.com/leeandyg/camtrap-tagger"

DEPLOYMENT_FIELDS = [
    "deploymentID", "locationID", "locationName", "latitude", "longitude",
    "coordinateUncertainty", "deploymentStart", "deploymentEnd", "setupBy", "cameraID",
    "cameraModel", "cameraDelay", "cameraHeight", "cameraDepth", "cameraTilt", "cameraHeading",
    "detectionDistance", "timestampIssues", "baitUse", "featureType", "habitat",
    "deploymentGroups", "deploymentTags", "deploymentComments",
]
MEDIA_FIELDS = [
    "mediaID", "deploymentID", "captureMethod", "timestamp", "filePath", "filePublic",
    "fileName", "fileMediatype", "exifData", "favorite", "mediaComments",
]
OBSERVATION_FIELDS = [
    "observationID", "deploymentID", "mediaID", "eventID", "eventStart", "eventEnd",
    "observationLevel", "observationType", "cameraSetupType", "scientificName", "count",
    "lifeStage", "sex", "behavior", "individualID", "individualPositionRadius",
    "individualPositionAngle", "individualSpeed", "bboxX", "bboxY", "bboxWidth", "bboxHeight",
    "classificationMethod", "classifiedBy", "classificationTimestamp",
    "classificationProbability", "observationTags", "observationComments",
]
RANKS = ["class", "order", "family", "genus", "species"]


def parse_timezone(value):
    """An IANA time zone name ("America/Los_Angeles") or a fixed offset ("-07:00")."""
    match = re.fullmatch(r"([+-])(\d{1,2}):?(\d{2})", value.strip())
    if match:
        sign, hours, minutes = match.groups()
        offset = timedelta(hours=int(hours), minutes=int(minutes))
        return timezone(-offset if sign == "-" else offset)
    try:
        return ZoneInfo(value.strip())
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError(f"unknown time zone '{value}' (use a name like America/Los_Angeles "
                         "or an offset like -07:00)") from None


def localize(timestamp, tz):
    """Camera-local ISO time → ISO time with the deployment's UTC offset."""
    return datetime.fromisoformat(timestamp).replace(tzinfo=tz).isoformat(timespec="seconds")


def scientific_name(ranks):
    """(scientific name, rank) from [class, order, family, genus, species], or (None, None)."""
    if ranks[4] and ranks[3]:
        return f"{ranks[3].capitalize()} {ranks[4]}", "species"
    for rank, name in reversed(list(zip(RANKS[:4], ranks[:4]))):
        if name:
            return name.capitalize(), rank
    return None, None


def media_id(relpath):
    return hashlib.sha1(relpath.encode("utf-8")).hexdigest()[:12]


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return round(value, 6)
    return value


def _write_table(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(row.get(k)) for k in fields})


def write_camtrap_dp(results, out_dir, source_root, tz, project_title, contributors=(),
                     sampling_design="opportunistic"):
    """Write a Camtrap DP package to out_dir. Returns a list of warnings about left-out data."""
    warnings = []
    meta = results["metadata"]
    model = meta.get("model", "").rstrip("/").split("/")
    classified_by = (f"SpeciesNet {model[-2] if len(model) >= 2 else meta.get('model')} "
                     f"(speciesnet {meta.get('speciesnet_version')}), MegaDetector v5a")
    classified_at = datetime.fromisoformat(meta["processed_at"]).astimezone().isoformat(timespec="seconds")

    # Deployments: only those with the fields Camtrap DP requires.
    deployments, skipped = {}, []
    for d in results["deployments"]:
        if None in (d["latitude"], d["longitude"], d["start"], d["end"]):
            skipped.append(d["deploymentID"])
            continue
        deployments[d["deploymentID"]] = {
            "deploymentID": d["deploymentID"],
            "locationName": d["locationName"],
            "latitude": d["latitude"],
            "longitude": d["longitude"],
            "deploymentStart": localize(d["start"], tz),
            "deploymentEnd": localize(d["end"], tz),
        }
    if skipped:
        warnings.append(f"Left out deployment(s) without coordinates and start/end in the sheet "
                        f"(and their images): {', '.join(skipped)}")

    # Media: images in an included deployment with a capture time.
    by_image = {}
    for e in results["predictions"]:
        by_image.setdefault(e["filepath"], []).append(e)
    media, untimed = {}, 0
    for path, entries in by_image.items():
        first = entries[0]
        if first.get("deployment_id") not in deployments or first["category"] == "error":
            continue
        if not first.get("timestamp"):
            untimed += 1
            continue
        relpath = os.path.relpath(path, source_root).replace(os.sep, "/")
        media[path] = {
            "mediaID": media_id(relpath),
            "deploymentID": first["deployment_id"],
            "captureMethod": "activityDetection",
            "timestamp": localize(first["timestamp"], tz),
            "filePath": relpath,
            "filePublic": False,
            "fileName": os.path.basename(path),
            "fileMediatype": (VIDEO_MEDIATYPES.get(os.path.splitext(path)[1].lower())
                              or mimetypes.guess_type(path)[0] or "image/jpeg"),
        }
    if untimed:
        warnings.append(f"Left out {untimed} file(s) without a capture time (Camtrap DP requires one)")

    # Observations: per detection (media level), then per species per event (event level).
    observations, ranks_by_label = [], {}
    for path, m in media.items():
        found = [e for e in by_image[path] if e["category"] not in ("blank", "error")]
        if not found:
            observations.append({
                "observationID": f"{m['mediaID']}_1", "deploymentID": m["deploymentID"],
                "mediaID": m["mediaID"], "eventID": by_image[path][0].get("event_id"),
                "eventStart": m["timestamp"], "eventEnd": m["timestamp"],
                "observationLevel": "media", "observationType": "blank",
                "classificationMethod": "machine", "classifiedBy": classified_by,
                "classificationTimestamp": classified_at,
            })
            continue
        for n, e in enumerate(found, start=1):
            ranks = label_ranks(e)
            ranks_by_label.setdefault(e["prediction"], ranks)
            name = scientific_name(ranks)[0] if e["category"] == "animal" else None
            is_video_detection = e.get("first_seen") is not None
            bbox = (None if is_video_detection else e.get("bbox")) or [None] * 4
            start, end = m["timestamp"], m["timestamp"]
            if is_video_detection:
                taken = datetime.fromisoformat(m["timestamp"])
                start = (taken + timedelta(seconds=e["first_seen"])).isoformat(timespec="seconds")
                end = (taken + timedelta(seconds=e["last_seen"])).isoformat(timespec="seconds")
            probability = e.get("species_conf") if e.get("species_conf") is not None else e.get("detector_conf")
            observations.append({
                "observationID": f"{m['mediaID']}_{n}", "deploymentID": m["deploymentID"],
                "mediaID": m["mediaID"], "eventID": e.get("event_id"),
                "eventStart": start, "eventEnd": end,
                "observationLevel": "media", "observationType": e["category"],
                "scientificName": name, "count": 1,
                "bboxX": bbox[0], "bboxY": bbox[1], "bboxWidth": bbox[2], "bboxHeight": bbox[3],
                "classificationMethod": "machine", "classifiedBy": classified_by,
                "classificationTimestamp": classified_at, "classificationProbability": probability,
            })

    for ev in results.get("events", []):
        if ev["deployment_id"] not in deployments:
            continue
        for n, o in enumerate(ev["observations"], start=1):
            ranks = ranks_by_label.get(o["label"], [""] * 5)
            observations.append({
                "observationID": f"{ev['event_id']}_{n}", "deploymentID": ev["deployment_id"],
                "eventID": ev["event_id"],
                "eventStart": localize(ev["start"], tz), "eventEnd": localize(ev["end"], tz),
                "observationLevel": "event", "observationType": o["category"],
                "scientificName": scientific_name(ranks)[0] if o["category"] == "animal" else None,
                "count": o["count"],
                "classificationMethod": "machine", "classifiedBy": classified_by,
                "classificationTimestamp": classified_at,
                "classificationProbability": o["confidence"],
            })

    os.makedirs(out_dir, exist_ok=True)
    used = {m["deploymentID"] for m in media.values()}
    deployment_rows = [d for d_id, d in deployments.items() if d_id in used]
    _write_table(os.path.join(out_dir, "deployments.csv"), DEPLOYMENT_FIELDS, deployment_rows)
    _write_table(os.path.join(out_dir, "media.csv"), MEDIA_FIELDS, media.values())
    _write_table(os.path.join(out_dir, "observations.csv"), OBSERVATION_FIELDS, observations)

    descriptor = package_descriptor(deployment_rows, observations, ranks_by_label, project_title,
                                    contributors, sampling_design)
    with open(os.path.join(out_dir, "datapackage.json"), "w", encoding="utf-8") as f:
        json.dump(descriptor, f, indent=2, ensure_ascii=False)
    return warnings


def package_descriptor(deployments, observations, ranks_by_label, project_title, contributors,
                       sampling_design):
    """datapackage.json contents: resources plus the metadata Camtrap DP requires."""
    resources = [{
        "name": name, "path": f"{name}.csv", "profile": "tabular-data-resource",
        "format": "csv", "mediatype": "text/csv", "encoding": "utf-8",
        "schema": f"{SCHEMA_BASE}/{name}-table-schema.json",
    } for name in ("deployments", "media", "observations")]

    lats = [d["latitude"] for d in deployments]
    lons = [d["longitude"] for d in deployments]
    if not deployments:
        spatial = None
    elif min(lats) == max(lats) and min(lons) == max(lons):
        spatial = {"type": "Point", "coordinates": [lons[0], lats[0]]}
    else:
        w, e, s, n = min(lons), max(lons), min(lats), max(lats)
        spatial = {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}

    taxa = {}
    for label, ranks in ranks_by_label.items():
        name, rank = scientific_name(ranks)
        if name and name not in taxa:
            taxon = {"scientificName": name, "taxonRank": rank}
            taxon.update({r: v.capitalize() for r, v in zip(RANKS[:4], ranks[:4]) if v})
            taxon["vernacularNames"] = {"eng": label}
            taxa[name] = taxon
    used_names = {o["scientificName"] for o in observations if o.get("scientificName")}

    slug = re.sub(r"[^a-z0-9]+", "-", project_title.lower()).strip("-") or "camtrap-tagger"
    return {
        "resources": resources,
        "profile": f"{SCHEMA_BASE}/camtrap-dp-profile.json",
        "name": slug,
        "id": str(uuid.uuid4()),
        "created": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "title": project_title,
        "description": ("Machine classifications from camtrap-tagger (SpeciesNet). Media filePath "
                        "values are relative to the original image folder; files are not public."),
        "contributors": list(contributors),
        "sources": [{"title": "camtrap-tagger", "path": REPOSITORY,
                     "version": version("camtrap-tagger")}],
        "project": {
            "title": project_title,
            "samplingDesign": sampling_design,
            "captureMethod": ["activityDetection"],
            "individualAnimals": False,
            "observationLevel": ["media", "event"],
        },
        "spatial": spatial,
        "temporal": {
            "start": min(d["deploymentStart"] for d in deployments)[:10] if deployments else None,
            "end": max(d["deploymentEnd"] for d in deployments)[:10] if deployments else None,
        },
        "taxonomic": [taxa[n] for n in sorted(taxa) if n in used_names],
    }


def parse_contributor(text):
    """"Name" or "Name <email>" → a Camtrap DP contributor."""
    match = re.fullmatch(r"\s*(.+?)\s*<([^>]+)>\s*", text)
    if match:
        return {"title": match.group(1), "email": match.group(2), "role": "contact"}
    return {"title": text.strip(), "role": "contact"}
