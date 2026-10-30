# camtrap-tagger

Tag every animal in a folder of camera-trap images: one species label per animal, boxes drawn on
the images, and empty or uncertain frames sorted out for human review. Built on Google's
[SpeciesNet](https://github.com/google/cameratrapai) (MegaDetector + an EfficientNetV2 species
classifier).

![Two American crows and a cottontail, each boxed and labeled separately](docs/images/sample_tagged.jpg)

*Each animal gets its own label. The rabbit is only partly visible, so the label falls back to
"rabbit and hare family" rather than guessing a species. Image: ENA24 dataset (see
[credits](#credits)).*

## What it does

1. **Finds** animals, people and vehicles with MegaDetector.
2. **Classifies each animal separately**, so a frame with a deer and a squirrel gets both labels.
3. **Applies SpeciesNet's decision rules per animal**: location-based species filtering
   (geofencing) and taxonomic fallback. A low-confidence species becomes its genus or family
   rather than a wrong guess.
4. **Rechecks empty frames.** Images with no detection are run through SpeciesNet's whole-image
   classifier, which catches some animals the detector missed (flagged `detector_missed`).
5. **Draws** color-coded boxes with detector and species confidence, placing labels so they
   don't overlap.
6. **Sorts** blank and unknown images into review folders.
7. **Writes one flat results file** with one entry per detection.

## How this relates to SpeciesNet

This started before SpeciesNet could classify more than one animal per image: SpeciesNet 5.0.3
gave one label per image, so this project cropped every animal and ran the decision logic per
crop. SpeciesNet 5.0.6 now classifies every detection itself, and camtrap-tagger uses that as its
default backend. The original crop method is kept as a second backend for comparison.

What camtrap-tagger adds on top of SpeciesNet:

- Whole-image recheck of frames where the detector found nothing
- Built-in visualization (SpeciesNet points to separate tools for this)
- Review folders for blank and unknown images
- A flat, per-detection results file instead of nested per-image output
- Input validation (SpeciesNet silently ignores a mistyped country code)

If you want a desktop app rather than a command-line tool, see
[AddaxAI](https://addaxdatascience.com/addaxai/), which also runs SpeciesNet.

## Install

Requires [uv](https://docs.astral.sh/uv/) (it installs the right Python version for you).

```bash
git clone https://github.com/leeandyg/camtrap-tagger.git
cd camtrap-tagger
uv sync
```

The first run downloads the SpeciesNet models from Kaggle (about 500 MB each, cached afterwards;
the whole-image model is only downloaded if some image has no detections).
Runs on CPU, Apple Silicon (MPS) or NVIDIA GPUs.

## Usage

```bash
uv run camtrap-tagger path/to/images --country USA --admin1-region CA
```

| Option | Default | Description |
|---|---|---|
| `--out DIR` | `<folder>_tagged` | Output folder |
| `--country CODE` | none | 3-letter ISO country code (e.g. `USA`, `KEN`); enables geofencing |
| `--admin1-region CODE` | none | US state code (e.g. `CA`); requires `--country USA` |
| `--backend {upstream,crop}` | `upstream` | How animals are classified (see below) |
| `--threshold N` | `0.1` | Minimum detector confidence for a detection to count |
| `--model NAME` | SpeciesNet v4.0.3a | Crop classifier model |
| `--no-recheck` | | Skip the whole-image recheck of empty frames |
| `--no-viz` | | Don't draw boxes |

Images are found recursively (jpg, png, tif, webp). Output mirrors the input's subfolders:

```
<folder>_tagged/
  results.json                 one entry per detection + summary
  speciesnet_predictions.json  raw SpeciesNet output, for debugging
  visualizations/              images with boxes and labels
  review/blank/                images with nothing found
  review/unknown/              images with at least one unidentified detection
  crops/                       per-animal crops (crop backend only)
```

### Backends

Both backends use the same detector and the same SpeciesNet decision rules. They differ in how a
detection becomes classifier input:

- **`upstream`**: SpeciesNet's built-in per-detection mode. Classifies every detection above the
  threshold, including people and vehicles, so it can correct a mislabeled box (for example, a
  low-confidence "vehicle" box that is really a person).
- **`crop`**: the original method. Classifies animal detections only, cropping each to 480×480
  with Lanczos resampling, and saves the crops for review.

Which one is more accurate hasn't been measured yet; an evaluation on labeled data is planned.

### Results format

Each entry in `results.json` is one detection:

```json
{
  "filepath": "/data/site3/IMG_0042.jpg",
  "detection_index": 2,
  "category": "animal",
  "prediction": "rabbit and hare family",
  "prediction_raw": "eastern cottontail",
  "species_conf": 0.67,
  "detector_conf": 0.78,
  "bbox": [0.564, 0.827, 0.112, 0.143],
  "rollup_status": "confidence_rollup",
  "prediction_source": "classifier+rollup_to_family",
  "taxonomy_5part": "mammalia;lagomorpha;leporidae;sylvilagus;floridanus"
}
```

- `category` is `animal`, `human`, `vehicle`, `blank`, `unknown` or `error` (unreadable image).
- `prediction` is the final label; `prediction_raw` is the classifier's top guess before
  geofencing and rollup. The `taxonomy_*` fields describe `prediction_raw`.
- `bbox` is `[x, y, width, height]` as fractions of the image size. It is `null` for animals
  found by the whole-image recheck, which also carry `"note": "detector_missed"`.

## Development

```bash
uv run pytest        # ~50 tests, a few seconds, no models needed
```

Code lives in `src/camtrap_tagger/`: `pipeline.py` (orchestration and blank recheck),
`backends.py`, `results.py`, `visualize.py`, `cli.py`.

## Credits

- **SpeciesNet** by Google, Apache-2.0. Gadot T, Istrate Ș, Kim H, Morris D, Beery S, Birch T,
  Ahumada J. *To crop or not to crop: Comparing whole-image and cropped classification on a large
  dataset of camera trap images.* IET Computer Vision, 2024.
- **MegaDetector** by Dan Morris and contributors,
  [github.com/agentmorris/MegaDetector](https://github.com/agentmorris/MegaDetector).
- **Sample image** from the [ENA24-detection](https://lila.science/datasets/ena24detection) dataset
  on LILA BC, released under the
  [Community Data License Agreement (permissive variant)](https://cdla.io/permissive-1-0/).
  Yousif H, Kays R, He Z. *Dynamic Programming Selection of Object Proposals for Sequence-Level
  Animal Species Classification in the Wild.* IEEE Transactions on Circuits and Systems for Video
  Technology, 2019.

## License

MIT. See [LICENSE](LICENSE).
