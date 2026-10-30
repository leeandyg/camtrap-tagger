import json
import os
import subprocess
import shutil
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont


# ============================================================================
# ENSEMBLE HELPER
# ============================================================================
HELPER_SCRIPT_CONTENT = '''"""
Helper script to run SpeciesNet ensemble inside the conda environment.

Usage:
    conda run -n speciesnet python ensemble_helper.py --input input.json --output output.json

Input JSON format:
{
    "model_name": "kaggle:google/speciesnet/pyTorch/v4.0.2a/1",
    "batches": [
        {
            "filepaths": ["path/to/image.jpg"],
            "classifier_results": {"path/to/image.jpg": {"classifications": {...}}},
            "detector_results": {"path/to/image.jpg": {"detections": [...]}},
            "geolocation_results": {"path/to/image.jpg": {"country": "USA"}}
        },
        ...
    ]
}

Output JSON format:
{
    "results": [
        {<ensemble output dict for batch 0>},
        {<ensemble output dict for batch 1>},
        ...
    ]
}
"""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Run SpeciesNet ensemble combine")
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    from speciesnet.ensemble import SpeciesNetEnsemble

    model_name = data.get("model_name", "kaggle:google/speciesnet/pyTorch/v4.0.2a/1")
    print(f"  Loading SpeciesNet Ensemble ({model_name})...")
    ensemble = SpeciesNetEnsemble(model_name=model_name)
    print("  - Ensemble loaded successfully")

    results = []
    batches = data.get("batches", [])
    for i, batch in enumerate(batches):
        try:
            output = ensemble.combine(
                filepaths=batch["filepaths"],
                classifier_results=batch["classifier_results"],
                detector_results=batch["detector_results"],
                geolocation_results=batch["geolocation_results"],
                partial_predictions=batch.get("partial_predictions", {}),
            )
            # output is a list; take first element (single-image batch)
            results.append(output[0] if output else {})
        except Exception as e:
            print(f"  Warning: Ensemble failed for batch {i}: {e}")
            results.append({})

    with open(args.output, "w") as f:
        json.dump({"results": results}, f, indent=2)

    print(f"  - Ensemble processed {len(results)} batches")


if __name__ == "__main__":
    main()
'''

HELPER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ensemble_helper.py")


def ensure_helper_script_exists():
    """Create the ensemble helper script if it doesn't exist."""
    if not os.path.exists(HELPER_SCRIPT):
        with open(HELPER_SCRIPT, 'w') as f:
            f.write(HELPER_SCRIPT_CONTENT)


def run_ensemble_subprocess(batches, output_dir):
    """
    Run ensemble.combine() via subprocess in the speciesnet conda environment.

    Args:
        batches: list of dicts, each with keys: filepaths, classifier_results,
                 detector_results, geolocation_results
        output_dir: directory for temp files

    Returns:
        list of ensemble output dicts (one per batch), or None on failure
    """
    # Ensure helper script exists (creates it if needed)
    ensure_helper_script_exists()
    
    input_path = os.path.join(output_dir, "temp_ensemble_input.json")
    output_path = os.path.join(output_dir, "temp_ensemble_output.json")

    input_data = {
        "model_name": "kaggle:google/speciesnet/pyTorch/v4.0.2a/1",
        "batches": batches,
    }

    with open(input_path, "w") as f:
        json.dump(input_data, f)

    cmd = (
        f'conda run -n speciesnet --no-capture-output python "{HELPER_SCRIPT}" '
        f'--input "{input_path}" --output "{output_path}"'
    )
    result = subprocess.run(cmd, shell=True)

    # Cleanup input
    try:
        os.remove(input_path)
    except OSError:
        pass

    if result.returncode != 0 or not os.path.exists(output_path):
        print("  Error: Ensemble subprocess failed.")
        return None

    with open(output_path, "r") as f:
        output_data = json.load(f)

    # Cleanup output
    try:
        os.remove(output_path)
    except OSError:
        pass

    return output_data.get("results", [])


def parse_taxonomy_label(label):
    """
    Parse a SpeciesNet taxonomy label.

    Returns:
        dict with keys:
            - taxonomy_uuid
            - taxonomy_5part (class;order;family;genus;species) padded to 5 parts
            - taxonomy_raw (original label string)
            - common_name (last non-empty rank)
    """
    result = {
        "taxonomy_uuid": None,
        "taxonomy_5part": None,
        "taxonomy_raw": label,
        "common_name": "unknown"
    }

    if not label:
        return result

    parts = label.split(';')
    if len(parts) >= 6:
        result["taxonomy_uuid"] = parts[0] or None
        tax_parts = parts[1:6]
    else:
        tax_parts = parts[:5]

    while len(tax_parts) < 5:
        tax_parts.append("")

    result["taxonomy_5part"] = ';'.join(tax_parts)

    for part in reversed(tax_parts):
        if part:
            result["common_name"] = part
            break

    return result


def parse_ensemble_prediction(prediction):
    """
    Parse ensemble prediction into a clean common name and status flags.

    Returns:
        dict with keys:
            - clean_name
            - raw
            - is_blank
            - is_unknown
    """
    raw = prediction
    if raw is None:
        return {
            "clean_name": "unknown",
            "raw": None,
            "is_blank": False,
            "is_unknown": False
        }

    raw_str = raw if isinstance(raw, str) else str(raw)
    clean_name = raw_str.split(";")[-1].strip() if ";" in raw_str else raw_str.strip()

    is_blank = raw_str == "blank" or raw_str.endswith(";blank")
    is_unknown = (raw_str == "unknown" or raw_str.endswith(";unknown") or
                  "no cv result" in raw_str)

    if is_unknown:
        clean_name = "unknown"
    elif is_blank:
        clean_name = "blank"

    return {
        "clean_name": clean_name,
        "raw": raw_str,
        "is_blank": is_blank,
        "is_unknown": is_unknown
    }


def derive_rollup_status(clean_prediction, prediction_source, raw_common_name=None):
    """Infer a rollup status string from ensemble output."""
    if clean_prediction == "blank":
        return "blank"
    if clean_prediction == "unknown":
        return "unknown"
    if clean_prediction == "animal":
        return "rolled_to_animal"

    source_lower = (prediction_source or "").lower()
    if "geofence" in source_lower:
        return "geofence_rollup" if "rollup" in source_lower else "geofence_valid"
    if "rollup" in source_lower:
        return "confidence_rollup"
    if source_lower == "detector":
        return "detector"
    if source_lower == "classifier":
        return "valid"

    if raw_common_name and clean_prediction != raw_common_name:
        return "rolled_up"

    return "not_applicable"


def update_rollup_summary(summary, rollup_status):
    if rollup_status == "confidence_rollup":
        summary["confidence_rollups"] += 1
    if rollup_status == "geofence_rollup":
        summary["geofencing_rollups"] += 1


# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

# Color scheme for different detection categories (RGB)
COLORS = {
    "animal": (0, 255, 0),       # Green
    "human": (255, 165, 0),      # Orange
    "vehicle": (0, 191, 255),    # Deep Sky Blue
    "unknown": (255, 0, 255),    # Magenta
}

# Priority for drawing order (lower = drawn first/behind, higher = drawn last/on top)
CATEGORY_PRIORITY = {
    "vehicle": 0,
    "human": 1,
    "animal": 2,
    "unknown": 2,
    "blank": -1,
}


def get_top_species_info(pred_entry):
    """
    Safely extracts top species info from SpeciesNet classifications.
    
    Returns:
        dict with keys:
            - prediction: Common name (e.g., "white-tailed deer")
            - full_classification: 5-part taxonomy string
            - taxonomy_uuid: UUID from classifier label
            - taxonomy_5part: 5-part taxonomy string (padded)
            - taxonomy_raw: original classifier label string
            - species_conf: Confidence score
    """
    class_data = pred_entry.get('classifications', {})
    classes = class_data.get('classes', [])
    scores = class_data.get('scores', [])
    
    result = {
        "prediction": "unknown",
        "full_classification": None,
        "taxonomy_uuid": None,
        "taxonomy_5part": None,
        "taxonomy_raw": None,
        "species_conf": 0.0
    }
    
    if classes:
        taxonomy = parse_taxonomy_label(classes[0])
        result["prediction"] = taxonomy["common_name"]
        result["taxonomy_uuid"] = taxonomy["taxonomy_uuid"]
        result["taxonomy_5part"] = taxonomy["taxonomy_5part"]
        result["taxonomy_raw"] = taxonomy["taxonomy_raw"]
        result["full_classification"] = taxonomy["taxonomy_5part"] # For backward compatibility

    
    if scores:
        result["species_conf"] = scores[0]
    
    return result


def get_user_input():
    """Get source folder and optional country code from user."""
    folder_input = input("Drag and drop your image folder: ").strip()
    source_folder = folder_input.replace("\'", "").replace('"', "")
    
    if not os.path.isdir(source_folder):
        print(f"Error: '{source_folder}' is not a valid directory.")
        return None, None
    
    country_code = input("Enter 3-letter country code for geofencing (or press Enter to skip): ").strip().upper()
    country_code = country_code if len(country_code) == 3 else None
    
    return source_folder, country_code


def setup_output_folders(source_folder):
    """Create output folder structure adjacent to source folder."""
    parent_dir = os.path.dirname(os.path.abspath(source_folder))
    source_name = os.path.basename(os.path.abspath(source_folder))
    output_dir = os.path.join(parent_dir, f"{source_name}_speciesnet_outputs")
    
    paths = {
        "output_dir": output_dir,
        "detector_json_dir": os.path.join(output_dir, "detector_json"),
        "crops_dir": os.path.join(output_dir, "crops"),
        "blanks_dir": os.path.join(output_dir, "blanks"),
        "unknowns_dir": os.path.join(output_dir, "unknowns"),
        "visualizations_dir": os.path.join(output_dir, "visualizations"),
        "final_results_dir": os.path.join(output_dir, "final_results"),
    }
    
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    
    return paths


def run_detector(source_folder, detector_json_path):
    """Run SpeciesNet in detector-only mode."""
    print("\n[Step 1/7] Running MegaDetector (detector-only mode)...")
    cmd = (f'conda run -n speciesnet --no-capture-output python -m speciesnet.scripts.run_model --detector_only '
           f'--folders "{source_folder}" --predictions_json "{detector_json_path}"')
    subprocess.run(cmd, shell=True)
    
    if not os.path.exists(detector_json_path):
        print("Error: Detector failed to produce output.")
        return None
    
    with open(detector_json_path, 'r') as f:
        return json.load(f)


def process_detections(detector_data, blanks_dir, conf_threshold=0.1):
    """
    Process ALL detections from detector results at the detection level.
    
    Returns:
    - blank_images: List of filepaths with no detections
    - human_vehicle_detections: List of dicts with {filepath, category, label, conf, bbox, detection_index}
    - animal_entries: List of (filepath, [animal_detections_with_index]) for cropping
    """
    print("\n[Step 2/7] Processing all detections...")
    
    blank_images = []
    human_vehicle_detections = []
    animal_entries = []
    
    total_humans = 0
    total_vehicles = 0
    total_animals = 0
    
    for entry in detector_data.get('predictions', []):
        filepath = entry.get('filepath')
        detections = entry.get('detections', [])
        
        # Filter detections above confidence threshold
        valid_detections = [d for d in detections if d.get('conf', 0) > conf_threshold]
        
        if not valid_detections:
            # No valid detections - mark as blank for verification
            blank_images.append(filepath)
            continue
        
        # Separate detections by category, keeping track of detection index
        animal_dets_with_index = []
        detection_index = 0
        
        for det in valid_detections:
            category = det.get('category')
            
            if category == '1':  # Animal
                animal_dets_with_index.append({
                    'detection_index': detection_index,
                    'bbox': det.get('bbox'),
                    'conf': det.get('conf'),
                    'label': det.get('label', 'animal')
                })
                total_animals += 1
            elif category == '2':  # Human
                human_vehicle_detections.append({
                    'filepath': filepath,
                    'category': 'human',
                    'label': 'human',
                    'conf': det.get('conf'),
                    'bbox': det.get('bbox'),
                    'detection_index': detection_index
                })
                total_humans += 1
            elif category == '3':  # Vehicle
                human_vehicle_detections.append({
                    'filepath': filepath,
                    'category': 'vehicle',
                    'label': 'vehicle',
                    'conf': det.get('conf'),
                    'bbox': det.get('bbox'),
                    'detection_index': detection_index
                })
                total_vehicles += 1
            
            detection_index += 1
        
        # If there are animal detections for this image, add to animal_entries
        if animal_dets_with_index:
            animal_entries.append((filepath, animal_dets_with_index))
    
    # Copy blank images to blanks folder for verification
    print(f"  - Copying {len(blank_images)} potential blank images for verification...")
    for filepath in blank_images:
        try:
            dest = os.path.join(blanks_dir, os.path.basename(filepath))
            shutil.copy2(filepath, dest)
        except Exception as e:
            print(f"  Warning: Could not copy {filepath}: {e}")
    
    print(f"  - Found: {len(blank_images)} blanks, {total_humans} humans, {total_vehicles} vehicles, {total_animals} animals")
    print(f"  - Images with animals: {len(animal_entries)}")
    
    return blank_images, human_vehicle_detections, animal_entries


def generate_animal_crops(animal_entries, crops_dir):
    """Generate 480x480 crops for all animal detections."""
    print("\n[Step 3/7] Generating 480x480 crops for animal detections...")
    
    crop_metadata_map = {}
    crop_count = 0
    
    for filepath, detections in animal_entries:
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        
        try:
            with Image.open(filepath) as img:
                img_w, img_h = img.size
                for det in detections:
                    det_index = det['detection_index']
                    # Bbox format: (xmin, ymin, width, height) normalized
                    xmin, ymin, w, h = det['bbox']
                    left = xmin * img_w
                    top = ymin * img_h
                    right = (xmin + w) * img_w
                    bottom = (ymin + h) * img_h
                    
                    # Crop and resize to 480x480
                    crop = img.crop((left, top, right, bottom)).resize((480, 480), Image.Resampling.LANCZOS)
                    crop_filename = f"{base_name}_det{det_index}.jpg"
                    crop_path = os.path.join(crops_dir, crop_filename)
                    crop.save(crop_path, quality=95)
                    
                    crop_metadata_map[crop_filename] = {
                        "original_filepath": filepath,
                        "bbox": det['bbox'],
                        "detector_conf": det['conf'],
                        "detection_index": det_index
                    }
                    crop_count += 1
        except Exception as e:
            print(f"  Warning: Error cropping {filepath}: {e}")
    
    print(f"  - Generated {crop_count} crops")
    return crop_metadata_map


def run_crop_classifier(crops_dir, output_json_path):
    """Run SpeciesNet v4.0.2a classifier on animal crops."""
    print("\n[Step 4/7] Running SpeciesNet v4.0.2a classifier on animal crops...")
    
    # Check if there are any crops to process
    crop_files = [f for f in os.listdir(crops_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not crop_files:
        print("  - No crops to classify")
        return None
    
    cmd = (f'yes y | conda run -n speciesnet --no-capture-output python -m speciesnet.scripts.run_model --classifier_only '
           f'--folders "{crops_dir}" --predictions_json "{output_json_path}" '
           f'--model "kaggle:google/speciesnet/pyTorch/v4.0.2a/1"')
    subprocess.run(cmd, shell=True)
    
    if os.path.exists(output_json_path):
        with open(output_json_path, 'r') as f:
            return json.load(f)
    return None


def run_blank_classifier(blanks_dir, output_json_path):
    """Run SpeciesNet v4.0.2b classifier on potential blank images."""
    print("\n[Step 5/7] Verifying blank images with v4.0.2b classifier...")
    
    # Check if there are any blanks to process
    blank_files = [f for f in os.listdir(blanks_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not blank_files:
        print("  - No blank images to verify")
        return None
    
    cmd = (f'yes y | conda run -n speciesnet --no-capture-output python -m speciesnet.scripts.run_model --classifier_only '
           f'--folders "{blanks_dir}" --predictions_json "{output_json_path}" '
           f'--model "kaggle:google/speciesnet/pyTorch/v4.0.2b/1"')
    subprocess.run(cmd, shell=True)
    
    if os.path.exists(output_json_path):
        with open(output_json_path, 'r') as f:
            return json.load(f)
    return None


def move_unknown_crops(crop_classifier_results, crop_metadata_map, crops_dir, unknowns_dir):
    """Move crops classified as 'unknown' to the unknowns folder."""
    unknown_count = 0
    
    if not crop_classifier_results:
        return unknown_count
    
    for pred in crop_classifier_results.get('predictions', []):
        crop_fname = os.path.basename(pred['filepath'])
        species_info = get_top_species_info(pred)
        
        if species_info["prediction"] == "unknown" and crop_fname in crop_metadata_map:
            src_path = os.path.join(crops_dir, crop_fname)
            dst_path = os.path.join(unknowns_dir, crop_fname)
            try:
                if os.path.exists(src_path):
                    shutil.move(src_path, dst_path)
                    unknown_count += 1
            except Exception as e:
                print(f"  Warning: Could not move unknown crop {crop_fname}: {e}")
    
    return unknown_count


def merge_results(blank_images, blank_classifier_results, human_vehicle_detections, 
                  crop_classifier_results, crop_metadata_map, unknowns_dir,
                  source_folder, country_code):
    """Merge all result streams into unified output with per-detection entries."""
    print("\n[Step 6/7] Merging all results...")

    final_predictions = []
    summary = {
        "total_images": 0,
        "total_detections": 0,
        "animals": 0,
        "blanks": 0,
        "humans": 0,
        "vehicles": 0,
        "unknowns": 0,
        "detector_missed": 0,
        "geofencing_rollups": 0,
        "confidence_rollups": 0
    }

    geofencing_applied = bool(country_code)

    # Create a mapping from blank filename to original filepath
    blank_filepath_map = {os.path.basename(fp): fp for fp in blank_images}

    # 1. Add ALL Human/Vehicle detections (each gets its own entry)
    for det in human_vehicle_detections:
        final_predictions.append({
            "filepath": det["filepath"],
            "detection_index": det["detection_index"],
            "category": det["category"],
            "prediction": det["label"],
            "prediction_raw": det["label"],
            "full_classification": None,
            "taxonomy_uuid": None,
            "taxonomy_5part": None,
            "taxonomy_raw": None,
            "rollup_status": "not_applicable",
            "prediction_source": "detector",
            "species_conf": None,
            "bbox": det["bbox"],
            "detector_conf": det["conf"],
            "crop_filepath": None
        })
        if det["category"] == "human":
            summary["humans"] += 1
        else:
            summary["vehicles"] += 1

    # ---- Build ensemble batches for blanks and animal crops ----
    ensemble_batches = []   # list of batch dicts for the subprocess
    batch_metadata = []     # parallel list tracking what each batch represents

    # 2. Prepare blank ensemble batches
    if blank_classifier_results:
        for pred in blank_classifier_results.get('predictions', []):
            crop_fname = os.path.basename(pred['filepath'])
            original_filepath = blank_filepath_map.get(crop_fname, pred['filepath'])

            species_info = get_top_species_info(pred)
            geo = {"country": country_code} if country_code else {}

            ensemble_batches.append({
                "filepaths": [original_filepath],
                "classifier_results": {
                    original_filepath: {"classifications": pred.get("classifications", {})}
                },
                "detector_results": {original_filepath: {"detections": []}},
                "geolocation_results": {original_filepath: geo},
            })
            batch_metadata.append({
                "type": "blank",
                "original_filepath": original_filepath,
                "species_info": species_info,
                "pred": pred,
            })

    # 3. Prepare animal crop ensemble batches
    if crop_classifier_results:
        for pred in crop_classifier_results.get('predictions', []):
            crop_fname = os.path.basename(pred['filepath'])
            if crop_fname not in crop_metadata_map:
                continue

            meta = crop_metadata_map[crop_fname]
            species_info = get_top_species_info(pred)
            geo = {"country": country_code} if country_code else {}

            ensemble_batches.append({
                "filepaths": [meta['original_filepath']],
                "classifier_results": {
                    meta['original_filepath']: {"classifications": pred.get("classifications", {})}
                },
                "detector_results": {meta['original_filepath']: {"detections": [{
                    "category": "1",
                    "label": "animal",
                    "conf": meta['detector_conf'],
                    "bbox": meta['bbox']
                }]}},
                "geolocation_results": {meta['original_filepath']: geo},
            })
            batch_metadata.append({
                "type": "animal",
                "crop_fname": crop_fname,
                "meta": meta,
                "species_info": species_info,
                "pred": pred,
            })

    # ---- Run ensemble subprocess ----
    ensemble_results = None
    if ensemble_batches:
        ensemble_results = run_ensemble_subprocess(ensemble_batches, os.path.dirname(source_folder))

    if ensemble_results is None and ensemble_batches:
        print("  Error: Ensemble subprocess failed. Aborting merge.")
        return None

    # ---- Process ensemble results ----
    if ensemble_results:
        for i, ens_output in enumerate(ensemble_results):
            bm = batch_metadata[i]

            parsed_prediction = parse_ensemble_prediction(
                ens_output.get("prediction", "unknown")
            )
            final_prediction = parsed_prediction["clean_name"]
            prediction_score = ens_output.get("prediction_score")
            prediction_source = ens_output.get("prediction_source")
            species_info = bm["species_info"]
            rollup_status = derive_rollup_status(
                final_prediction, prediction_source, species_info["prediction"]
            )

            if bm["type"] == "blank":
                entry = {
                    "filepath": bm["original_filepath"],
                    "detection_index": None,
                    "category": "blank",
                    "prediction": final_prediction,
                    "prediction_raw": species_info["prediction"],
                    "ensemble_prediction_raw": parsed_prediction["raw"],
                    "full_classification": species_info["taxonomy_5part"],
                    "taxonomy_uuid": species_info["taxonomy_uuid"],
                    "taxonomy_5part": species_info["taxonomy_5part"],
                    "taxonomy_raw": species_info["taxonomy_raw"],
                    "rollup_status": rollup_status,
                    "prediction_source": prediction_source,
                    "species_conf": prediction_score,
                    "bbox": None,
                    "detector_conf": None,
                    "crop_filepath": None
                }

                if parsed_prediction["is_blank"]:
                    summary["blanks"] += 1
                elif parsed_prediction["is_unknown"]:
                    entry["category"] = "unknown"
                    summary["unknowns"] += 1
                else:
                    entry["category"] = "animal"
                    entry["note"] = "detector_missed"
                    summary["detector_missed"] += 1
                    summary["animals"] += 1

                update_rollup_summary(summary, rollup_status)
                final_predictions.append(entry)

            elif bm["type"] == "animal":
                meta = bm["meta"]
                crop_fname = bm["crop_fname"]

                crop_path = bm["pred"]['filepath']
                unknown_path = os.path.join(unknowns_dir, crop_fname)
                if os.path.exists(unknown_path):
                    crop_path = unknown_path

                category = "animal"
                if parsed_prediction["is_blank"]:
                    category = "blank"
                elif parsed_prediction["is_unknown"]:
                    category = "unknown"

                entry = {
                    "filepath": meta['original_filepath'],
                    "detection_index": meta['detection_index'],
                    "category": category,
                    "prediction": final_prediction,
                    "prediction_raw": species_info["prediction"],
                    "ensemble_prediction_raw": parsed_prediction["raw"],
                    "full_classification": species_info["taxonomy_5part"],
                    "taxonomy_uuid": species_info["taxonomy_uuid"],
                    "taxonomy_5part": species_info["taxonomy_5part"],
                    "taxonomy_raw": species_info["taxonomy_raw"],
                    "rollup_status": rollup_status,
                    "prediction_source": prediction_source,
                    "species_conf": prediction_score,
                    "bbox": meta['bbox'],
                    "detector_conf": meta['detector_conf'],
                    "crop_filepath": crop_path
                }

                update_rollup_summary(summary, rollup_status)
                final_predictions.append(entry)

                if parsed_prediction["is_unknown"]:
                    summary["unknowns"] += 1
                elif parsed_prediction["is_blank"]:
                    summary["blanks"] += 1
                else:
                    summary["animals"] += 1

    # Handle blanks with no classifier results
    if not blank_classifier_results:
        for filepath in blank_images:
            final_predictions.append({
                "filepath": filepath,
                "detection_index": None,
                "category": "blank",
                "prediction": "blank",
                "prediction_raw": None,
                "full_classification": None,
                "taxonomy_uuid": None,
                "taxonomy_5part": None,
                "taxonomy_raw": None,
                "rollup_status": "not_applicable",
                "prediction_source": "detector_only",
                "species_conf": None,
                "bbox": None,
                "detector_conf": None,
                "crop_filepath": None
            })
            summary["blanks"] += 1

    summary["total_images"] = len(set(p["filepath"] for p in final_predictions))
    summary["total_detections"] = len(final_predictions)

    # Build final output structure
    output = {
        "metadata": {
            "source_folder": source_folder,
            "country_code": country_code,
            "geofencing_applied": geofencing_applied,
            "processed_at": datetime.now().isoformat()
        },
        "predictions": final_predictions,
        "summary": summary
    }

    return output


def get_font(size=16):
    """Get a font for drawing text, with fallback to default."""
    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSText.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for font_path in font_paths:
        try:
            return ImageFont.truetype(font_path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def check_overlap(rect1, rect2):
    """Check if two rectangles overlap. Rectangles are (x1, y1, x2, y2)."""
    x1_1, y1_1, x2_1, y2_1 = rect1
    x1_2, y1_2, x2_2, y2_2 = rect2
    return not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1)


def find_label_position(bbox_pixels, text_width, text_height, img_size, placed_labels, padding=4):
    """
    Find a non-overlapping position for the label.
    Tries: above bbox → inside top → below bbox → offset positions
    Returns: (x, y) position for the label
    """
    left, top, right, bottom = bbox_pixels
    img_w, img_h = img_size
    
    label_width = text_width + padding * 2
    label_height = text_height + padding * 2
    
    # Position candidates in order of preference
    candidates = [
        # Above bbox
        (left, top - label_height - 2),
        # Inside top of bbox
        (left + 2, top + 2),
        # Below bbox
        (left, bottom + 2),
        # Above bbox, shifted right
        (min(left + 20, right - label_width), top - label_height - 2),
        # Inside top, shifted
        (left + 20, top + 2),
        # Above bbox, at right edge
        (right - label_width, top - label_height - 2),
        # Below bbox, shifted
        (min(left + 20, right - label_width), bottom + 2),
    ]
    
    for x, y in candidates:
        # Ensure within image bounds
        x = max(0, min(x, img_w - label_width))
        y = max(0, min(y, img_h - label_height))
        
        label_rect = (x, y, x + label_width, y + label_height)
        
        # Check for overlaps with already placed labels
        overlaps = False
        for placed_rect in placed_labels:
            if check_overlap(label_rect, placed_rect):
                overlaps = True
                break
        
        if not overlaps:
            return (x, y), label_rect
    
    # Fallback: return first position even if it overlaps
    x, y = candidates[0]
    x = max(0, min(x, img_w - label_width))
    y = max(0, min(y, img_h - label_height))
    return (x, y), (x, y, x + label_width, y + label_height)


def draw_bbox_with_label(draw, img_size, bbox, label_lines, color, placed_labels, line_width=3):
    """Draw a bounding box with label text, avoiding overlap with other labels."""
    img_w, img_h = img_size
    xmin, ymin, w, h = bbox
    
    # Convert normalized coordinates to pixels
    left = int(xmin * img_w)
    top = int(ymin * img_h)
    right = int((xmin + w) * img_w)
    bottom = int((ymin + h) * img_h)
    
    # Draw bounding box
    for i in range(line_width):
        draw.rectangle([left - i, top - i, right + i, bottom + i], outline=color)
    
    # Get font and calculate text dimensions
    font = get_font(16)
    text = "\n".join(label_lines)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # Find non-overlapping position
    padding = 4
    (label_x, label_y), label_rect = find_label_position(
        (left, top, right, bottom), 
        text_width, text_height, 
        img_size, placed_labels, padding
    )
    
    # Draw label background
    draw.rectangle(
        [label_x, label_y, label_x + text_width + padding * 2, label_y + text_height + padding * 2],
        fill=color
    )
    
    # Draw label text (black text on colored background)
    draw.text((label_x + padding, label_y + padding), text, fill=(0, 0, 0), font=font)
    
    # Return the label rectangle for tracking
    return label_rect


def draw_text_overlay(draw, img_size, text_lines, color, placed_labels):
    """Draw text overlay for detector-missed cases (no bbox)."""
    font = get_font(20)
    text = "\n".join(text_lines)
    
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # Find position that doesn't overlap
    padding = 10
    img_w, img_h = img_size
    
    # Try different corner positions
    candidates = [
        (padding, padding),  # Top-left
        (img_w - text_width - padding * 3, padding),  # Top-right
        (padding, img_h - text_height - padding * 3),  # Bottom-left
        (img_w - text_width - padding * 3, img_h - text_height - padding * 3),  # Bottom-right
    ]
    
    label_width = text_width + padding
    label_height = text_height + padding
    
    chosen_pos = candidates[0]
    for x, y in candidates:
        label_rect = (x, y, x + label_width, y + label_height)
        overlaps = any(check_overlap(label_rect, placed) for placed in placed_labels)
        if not overlaps:
            chosen_pos = (x, y)
            break
    
    x, y = chosen_pos
    
    # Draw background rectangle
    draw.rectangle(
        [x, y, x + text_width + padding, y + text_height + padding],
        fill=color
    )
    
    # Draw text
    draw.text((x + padding // 2, y + padding // 2), text, fill=(0, 0, 0), font=font)
    
    return (x, y, x + label_width, y + label_height)


def format_label(detection):
    """Format the label text for a detection based on its category."""
    category = detection.get("category", "unknown")
    prediction = detection.get("prediction", "unknown")
    detector_conf = detection.get("detector_conf")
    species_conf = detection.get("species_conf")
    
    lines = []
    
    if category == "human":
        # Format: human: 92%
        if detector_conf is not None:
            lines.append(f"human: {int(detector_conf * 100)}%")
        else:
            lines.append("human")
    
    elif category == "vehicle":
        # Format: vehicle: 88%
        if detector_conf is not None:
            lines.append(f"vehicle: {int(detector_conf * 100)}%")
        else:
            lines.append("vehicle")
    
    elif category == "animal":
        # Format: animal: 95%
        #         white-tailed deer: 87%
        if detector_conf is not None:
            lines.append(f"animal: {int(detector_conf * 100)}%")
        else:
            lines.append("animal")
        
        if prediction == "unknown":
            lines.append("unknown")
        elif species_conf is not None:
            lines.append(f"{prediction}: {int(species_conf * 100)}%")
        else:
            lines.append(prediction)
    
    else:
        # Unknown or other
        lines.append(prediction)
        if species_conf is not None:
            lines.append(f"{int(species_conf * 100)}%")
    
    return lines


def sort_detections_for_drawing(detections):
    """
    Sort detections for drawing order:
    - By category priority (vehicles first, then humans, then animals)
    - Within each category, by confidence (lowest first, so highest is drawn last/on top)
    """
    def sort_key(det):
        category = det.get("category", "unknown")
        priority = CATEGORY_PRIORITY.get(category, 0)
        # Use detector_conf, fall back to species_conf, then 0
        conf = det.get("detector_conf") or det.get("species_conf") or 0
        return (priority, conf)
    
    return sorted(detections, key=sort_key)


def visualize_results(final_output, visualizations_dir):
    """Draw all detections on original images and save to visualizations folder."""
    print("\n[Step 7/7] Generating visualizations...")
    
    predictions = final_output.get("predictions", [])
    
    # Group predictions by filepath
    by_image = {}
    for pred in predictions:
        fp = pred.get("filepath")
        if not fp:
            continue
        # Skip blanks (they're in a separate folder)
        if pred.get("category") == "blank" and pred.get("prediction") == "blank":
            continue
        if fp not in by_image:
            by_image[fp] = []
        by_image[fp].append(pred)
    
    visualized_count = 0
    
    for filepath, detections in by_image.items():
        try:
            img = Image.open(filepath)
            # Convert to RGB if necessary (for PNG with alpha, etc.)
            if img.mode != "RGB":
                img = img.convert("RGB")
            draw = ImageDraw.Draw(img)
            
            # Sort detections: vehicles first, then humans, then animals
            # Within each category: lowest confidence first (so highest is on top)
            sorted_detections = sort_detections_for_drawing(detections)
            
            # Track placed label positions to avoid overlap
            placed_labels = []
            
            for det in sorted_detections:
                category = det.get("category", "unknown")
                bbox = det.get("bbox")
                
                # Get color for this category
                if det.get("prediction") == "unknown":
                    color = COLORS.get("unknown", (255, 0, 255))
                else:
                    color = COLORS.get(category, (128, 128, 128))
                
                # Format label
                label_lines = format_label(det)
                
                if bbox:
                    # Draw bbox with label
                    label_rect = draw_bbox_with_label(
                        draw, img.size, bbox, label_lines, color, placed_labels
                    )
                    placed_labels.append(label_rect)
                else:
                    # Detector missed - draw text overlay (no bbox)
                    label_rect = draw_text_overlay(
                        draw, img.size, label_lines, color, placed_labels
                    )
                    placed_labels.append(label_rect)
            
            # Save to visualizations folder
            output_filename = os.path.basename(filepath)
            output_path = os.path.join(visualizations_dir, output_filename)
            img.save(output_path, quality=95)
            visualized_count += 1
            
        except Exception as e:
            print(f"  Warning: Could not visualize {filepath}: {e}")
    
    print(f"  - Generated {visualized_count} visualized images")
    return visualized_count


def cleanup_temp_files(paths, detector_json_path, crop_json_path, blank_json_path):
    """Delete temporary JSON files but keep crops folder."""
    print("\nCleaning up temporary files...")
    
    for json_path in [detector_json_path, crop_json_path, blank_json_path]:
        if json_path and os.path.exists(json_path):
            try:
                os.remove(json_path)
            except Exception as e:
                print(f"  Warning: Could not remove {json_path}: {e}")
    
    # Remove detector_json directory if empty
    try:
        os.rmdir(paths["detector_json_dir"])
    except:
        pass


def process_wildlife_photos():
    """Main unified pipeline for processing wildlife camera trap images."""
    
    # Step 1: Get user input
    source_folder, country_code = get_user_input()
    if not source_folder:
        return
    
    # Step 2: Setup output folders
    paths = setup_output_folders(source_folder)
    print(f"\nOutput directory: {paths['output_dir']}")
    
    # Define paths for temporary JSON files
    detector_json_path = os.path.join(paths["detector_json_dir"], "detections.json")
    crop_json_path = os.path.join(paths["output_dir"], "temp_crop_classifier.json")
    blank_json_path = os.path.join(paths["output_dir"], "temp_blank_classifier.json")
    
    # Step 3: Run detector
    detector_data = run_detector(source_folder, detector_json_path)
    if not detector_data:
        return
    
    # Step 4: Process all detections (humans, vehicles, animals, blanks)
    blank_images, human_vehicle_detections, animal_entries = process_detections(
        detector_data, paths["blanks_dir"]
    )
    
    # Step 5: Generate crops for animal detections
    crop_metadata_map = generate_animal_crops(animal_entries, paths["crops_dir"])
    
    # Step 6: Run classifiers
    crop_classifier_results = run_crop_classifier(paths["crops_dir"], crop_json_path)
    blank_classifier_results = run_blank_classifier(paths["blanks_dir"], blank_json_path)
    
    # Step 7: Move unknown crops to unknowns folder
    unknown_count = move_unknown_crops(
        crop_classifier_results, crop_metadata_map, 
        paths["crops_dir"], paths["unknowns_dir"]
    )
    if unknown_count > 0:
        print(f"  - Moved {unknown_count} unknown crops to unknowns folder")
    
    # Step 8: Merge all results
    final_output = merge_results(
        blank_images, blank_classifier_results,
        human_vehicle_detections,
        crop_classifier_results, crop_metadata_map,
        paths["unknowns_dir"],
        source_folder, country_code
    )
    if not final_output:
        print("Error: Merge failed; skipping output/visualization steps.")
        return
    
    # Step 9: Save final results
    output_path = os.path.join(paths["final_results_dir"], "unified_results.json")
    with open(output_path, 'w') as f:
        json.dump(final_output, f, indent=2)
    
    # Step 10: Cleanup temp files (keep crops for verification)
    cleanup_temp_files(paths, detector_json_path, crop_json_path, blank_json_path)
    
    # Step 11: Generate visualizations (automatic)
    visualize_results(final_output, paths["visualizations_dir"])
    
    # Print summary
    summary = final_output["summary"]
    metadata = final_output["metadata"]
    print("\n" + "="*50)
    print("PROCESSING COMPLETE!")
    print("="*50)
    print(f"Total images processed: {summary['total_images']}")
    print(f"Total detections:       {summary['total_detections']}")
    print(f"  - Animals:            {summary['animals']}")
    print(f"  - Humans:             {summary['humans']}")
    print(f"  - Vehicles:           {summary['vehicles']}")
    print(f"  - Blanks:             {summary['blanks']}")
    print(f"  - Unknowns:           {summary['unknowns']}")
    if summary['detector_missed'] > 0:
        print(f"  - Detector missed:    {summary['detector_missed']} (found by classifier)")
    
    # Geofencing summary
    if metadata.get('geofencing_applied'):
        print(f"\nGeofencing ({metadata['country_code']}):")
        print(f"  - Species rolled up:  {summary['geofencing_rollups']}")
    elif metadata.get('country_code'):
        print(f"\nGeofencing: Not available (speciesnet.ensemble import failed)")
    print(f"  - Confidence rollups:  {summary['confidence_rollups']}")
    
    print(f"\nFinal results:   {output_path}")
    print(f"Visualizations:  {paths['visualizations_dir']}")
    print(f"Crops folder:    {paths['crops_dir']}")
    print(f"Unknowns folder: {paths['unknowns_dir']}")


if __name__ == "__main__":
    process_wildlife_photos()