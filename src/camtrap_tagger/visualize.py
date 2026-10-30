"""Draw color-coded boxes and confidence labels on the original images."""

import os
from functools import cache

from PIL import Image, ImageDraw, ImageFont

from camtrap_tagger.results import mirrored_path
from camtrap_tagger.video import read_frame

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


@cache
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
        except OSError:
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
        if not any(check_overlap(label_rect, placed) for placed in placed_labels):
            return (x, y), label_rect

    # Fallback: return first position even if it overlaps
    x, y = candidates[0]
    x = max(0, min(x, img_w - label_width))
    y = max(0, min(y, img_h - label_height))
    return (x, y), (x, y, x + label_width, y + label_height)


def draw_bbox_with_label(draw, img_size, bbox, label_lines, color, placed_labels, line_width=3,
                         font_size=16):
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
    font = get_font(font_size)
    text = "\n".join(label_lines)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Find non-overlapping position
    padding = font_size // 4
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


def draw_text_overlay(draw, img_size, text_lines, color, placed_labels, font_size=20):
    """Draw text overlay for detector-missed cases (no bbox)."""
    font = get_font(font_size)
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
        if not any(check_overlap(label_rect, placed) for placed in placed_labels):
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

    if category in ("human", "vehicle"):
        # Format: human: 92%
        if detector_conf is not None:
            lines.append(f"{category}: {int(detector_conf * 100)}%")
        else:
            lines.append(category)

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


def draw_image(filepath, detections, frame_number=None):
    """Return a copy of the image at filepath, or of one video frame, with detections drawn on it."""
    img = Image.open(filepath) if frame_number is None else read_frame(filepath, frame_number)
    # Convert to RGB if necessary (for PNG with alpha, etc.)
    if img.mode != "RGB":
        img = img.convert("RGB")
    draw = ImageDraw.Draw(img)

    # Scale text and lines with image width so labels stay readable on large frames
    font_size = max(16, img.width // 64)
    line_width = max(3, img.width // 400)

    # Track placed label positions to avoid overlap
    placed_labels = []

    for det in sort_detections_for_drawing(detections):
        if det.get("prediction") == "unknown":
            color = COLORS["unknown"]
        else:
            color = COLORS.get(det.get("category"), (128, 128, 128))

        label_lines = format_label(det)
        if det.get("bbox"):
            label_rect = draw_bbox_with_label(draw, img.size, det["bbox"], label_lines, color,
                                              placed_labels, line_width, font_size)
        else:
            # Detector missed - draw text overlay (no bbox)
            label_rect = draw_text_overlay(draw, img.size, label_lines, color, placed_labels,
                                           font_size + font_size // 4)
        placed_labels.append(label_rect)

    return img


def visualize_results(results, source_root, visualizations_dir):
    """Draw all non-blank detections and save them, mirroring the source folder layout.

    A video gets one JPEG per frame its detections came from: clip.mp4_frame000360.jpg.
    """
    by_image = {}
    for pred in results["predictions"]:
        # Blanks go to the review folder instead
        if pred["category"] in ("blank", "error"):
            continue
        by_image.setdefault((pred["filepath"], pred.get("frame_number")), []).append(pred)

    visualized_count = 0
    for (filepath, frame_number), detections in by_image.items():
        try:
            img = draw_image(filepath, detections, frame_number)
            output_path = mirrored_path(filepath, source_root, visualizations_dir)
            if frame_number is not None:
                output_path += f"_frame{frame_number:06d}.jpg"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            img.save(output_path, quality=95)
            visualized_count += 1
        except Exception as e:
            print(f"  Warning: Could not visualize {filepath}: {e}")

    return visualized_count
