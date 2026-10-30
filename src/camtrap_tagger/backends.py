"""Classification backends.

Both backends run MegaDetector, classify animal detections one at a time, and send
each classification through SpeciesNet's ensemble (geofencing + taxonomic rollup).
They differ only in how a detection becomes classifier input:

- "upstream": SpeciesNet's built-in per-detection mode (in speciesnet 5.0.6, not 5.0.3).
  Crops every detection above the threshold (humans and vehicles too) with torch.
- "crop": this project's original method, written before upstream supported
  per-detection classification. Crops animal detections only, resized to 480x480
  with Lanczos resampling, optionally saving each crop for review.
"""

import os

from PIL import Image
from speciesnet import SpeciesNet
from speciesnet.constants import NUM_CLASSIFICATIONS_FOR_ENSEMBLE
from speciesnet.geolocation import find_admin1_region
from speciesnet.utils import BBox, build_classifier_result, load_rgb_image
from tqdm import tqdm

CROP_SIZE = 480
FULL_FRAME = BBox(0.0, 0.0, 1.0, 1.0)


class UpstreamBackend:
    name = "upstream"

    def __init__(self, model_name):
        self.net = SpeciesNet(model_name, components="all")
        self.ensemble = self.net.ensemble

    def predict(self, filepaths, country=None, admin1_region=None, threshold=0.1):
        out = self.net.predict(
            filepaths=filepaths,
            country=country,
            admin1_region=admin1_region,
            detection_threshold_for_classification=threshold,
            progress_bars=True,
        )
        return out["predictions"]


class CropBackend:
    name = "crop"

    def __init__(self, model_name, crops_dir=None, source_root=None):
        self.net = SpeciesNet(model_name, components="all")
        self.ensemble = self.net.ensemble
        self.crops_dir = crops_dir
        self.source_root = source_root

    def predict(self, filepaths, country=None, admin1_region=None, threshold=0.1):
        detector_results = {
            p["filepath"]: p
            for p in self.net.detect(filepaths=filepaths, progress_bars=True)["predictions"]
        }

        classifier_results = {}
        for filepath in tqdm(filepaths, desc="Classifying crops"):
            detections = detector_results[filepath].get("detections")
            img = load_rgb_image(filepath) if detections is not None else None
            if img is None:
                classifier_results[filepath] = {"filepath": filepath, "failures": ["CLASSIFIER"]}
                continue

            indices = [
                i for i, det in enumerate(detections)
                if det["category"] == "1" and det["conf"] >= threshold
            ]
            inputs = []
            for i in indices:
                crop = crop_detection(img, detections[i]["bbox"])
                if self.crops_dir:
                    crop.save(self.crop_path(filepath, i), quality=95)
                inputs.extend(self.net.classifier.preprocess(crop, bboxes=[FULL_FRAME]))
            raw = self.net.classifier.predict(
                filepath, inputs, num_classes=NUM_CLASSIFICATIONS_FOR_ENSEMBLE
            )
            classifier_results[filepath] = build_classifier_result(raw, detections, indices, False)

        geolocation = {"country": country, "admin1_region": find_admin1_region(country, admin1_region)}
        predictions = self.ensemble.combine(
            filepaths=filepaths,
            classifier_results=classifier_results,
            detector_results=detector_results,
            geolocation_results={fp: geolocation for fp in filepaths},
        )

        if self.crops_dir:
            for pred in predictions:
                for i, det in enumerate(pred.get("detections") or []):
                    if "raw_classifications" in det:
                        det["crop_filepath"] = self.crop_path(pred["filepath"], i)
        return predictions

    def crop_path(self, filepath, detection_index):
        rel = os.path.relpath(filepath, self.source_root) if self.source_root else os.path.basename(filepath)
        path = os.path.join(self.crops_dir, f"{os.path.splitext(rel)[0]}_det{detection_index}.jpg")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path


def crop_detection(img, bbox):
    """Crop a normalized (xmin, ymin, width, height) bbox and resize it to CROP_SIZE square."""
    xmin, ymin, w, h = bbox
    box = (xmin * img.width, ymin * img.height, (xmin + w) * img.width, (ymin + h) * img.height)
    return img.crop(box).resize((CROP_SIZE, CROP_SIZE), Image.Resampling.LANCZOS)
