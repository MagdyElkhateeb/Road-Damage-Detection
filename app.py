"""Gradio demo for the fine-tuned YOLO26s road damage detector.

Place the trained YOLO26s checkpoint at ./best.pt, next to this file.
The detector's class names are read directly from the checkpoint.
"""

from collections import Counter
from pathlib import Path

import gradio as gr
from PIL import Image
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = ROOT / "best.pt"
EXAMPLES_DIR = ROOT / "Test Images"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

if not WEIGHTS_PATH.is_file():
    raise FileNotFoundError(
        f"Model weights not found: {WEIGHTS_PATH}\n"
        "Copy the trained YOLO26s best.pt into the same folder as app.py."
    )

# Load the trained checkpoint once rather than on every prediction.
model = YOLO(str(WEIGHTS_PATH))


def detect_damage(image: Image.Image | None, confidence: float):
    """Return an annotated image, summary, and per-detection table."""
    if image is None:
        return None, "Please upload a road image first.", []

    # A PIL input avoids assumptions about the uploaded image's filename.
    result = model.predict(
        source=image.convert("RGB"),
        imgsz=640,
        conf=float(confidence),
        verbose=False,
    )[0]

    # Ultralytics plot() returns a BGR NumPy array; Gradio/PIL expect RGB.
    annotated_rgb = Image.fromarray(result.plot()[..., ::-1])

    rows = []
    counts = Counter()
    if result.boxes is not None:
        for cls_id, conf, box in zip(
            result.boxes.cls.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.xyxy.tolist(),
        ):
            label = result.names[int(cls_id)]
            counts[label] += 1
            coordinates = ", ".join(str(round(v)) for v in box)
            rows.append([label, round(conf, 3), coordinates])

    if not rows:
        summary = "**Detections: 0** — No road damage found at this confidence threshold."
    else:
        breakdown = ", ".join(f"{label}: {count}" for label, count in sorted(counts.items()))
        summary = f"**Detections: {len(rows)}**  |  {breakdown}"

    return annotated_rgb, summary, rows


def get_example_images():
    """Discover local sample images without assuming particular filenames."""
    if not EXAMPLES_DIR.is_dir():
        return []
    return [
        [str(path)]
        for path in sorted(EXAMPLES_DIR.iterdir(), key=lambda p: p.name.lower())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


with gr.Blocks(title="Road Damage Detection | YOLO26s") as demo:
    gr.Markdown(
        "# Road Damage Detection\n"
        "Upload a road image to identify damage using the **fine-tuned YOLO26s** model. "
        "Detected regions are shown with bounding boxes, class labels, and confidence scores."
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Input road image",
                type="pil",
                image_mode="RGB",
                sources=["upload", "webcam", "clipboard"],
            )
            confidence_slider = gr.Slider(
                minimum=0.05,
                maximum=0.95,
                value=0.25,
                step=0.05,
                label="Confidence threshold",
            )
            detect_button = gr.Button("Detect road damage", variant="primary")
        with gr.Column():
            output_image = gr.Image(label="Detection result", format="png")
            summary_output = gr.Markdown("Upload an image and select **Detect road damage**.")

    detail_table = gr.Dataframe(
        headers=["Damage class", "Confidence", "Box: x1, y1, x2, y2"],
        datatype=["str", "number", "str"],
        label="Detected objects",
        interactive=False,
    )

    detect_button.click(
        fn=detect_damage,
        inputs=[input_image, confidence_slider],
        outputs=[output_image, summary_output, detail_table],
    )

    examples = get_example_images()
    if examples:
        gr.Examples(examples=examples, inputs=input_image, label="Try a sample image")

    gr.Markdown(
        "*Research demonstration. Predictions may miss damage or produce false positives; "
        "human review is recommended for operational decisions.*"
    )


if __name__ == "__main__":
    demo.launch()
