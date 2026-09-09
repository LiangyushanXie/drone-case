"""Full-resolution four-column overlays from already-computed detections."""

ARM_NAMES = ("baseline", "resolution", "sliced")
COLORS = {"tp": "#ff9d2e", "fp": "#ef5350", "ignored": "#89939e"}


def percent(value):
    return "--" if value is None else f"{value * 100:.1f}%"


def render_comparison(image_path, records, settings, destination):
    from PIL import Image, ImageDraw, ImageFont

    with Image.open(image_path) as image:
        original = image.convert("RGB")
    width, height = original.size
    font_size = max(12, width // 90)
    font = ImageFont.load_default(size=font_size)
    score_font = ImageFont.load_default(size=max(10, width // 130))
    header = 3 * (font_size + 9)
    gap = 12
    canvas = Image.new("RGB", (4 * width + 3 * gap, height + header), "#101820")
    headings = [
        "Ground truth: car",
        "Baseline: whole image 640",
        f"Whole image {settings['whole_image_size']}",
        f"Tiles {settings['tile_size']} / input {settings['tile_input_size']}",
    ]
    reference = records["baseline"]
    for column in range(4):
        panel = original.copy()
        draw = ImageDraw.Draw(panel)
        if column == 0:
            for region in reference["ignored_regions"]:
                if region["valid_geometry"]:
                    draw.rectangle(region["xyxy"], outline=COLORS["ignored"], width=2)
            for box in reference["ground_truth"]:
                draw.rectangle(box["xyxy"], outline="#18c77a", width=2)
            evaluation = reference["evaluation"]
            detail = f"GT={evaluation['raw_gt']} | evaluated={evaluation['evaluated_gt']}"
            legend = "Green: original car labels | gray: ignore areas"
        else:
            record = records[ARM_NAMES[column - 1]]
            evaluation = record["evaluation"]
            for box, match in zip(record["predictions"], evaluation["prediction_states"]):
                color = COLORS[match["status"]]
                draw.rectangle(box["xyxy"], outline=color, width=2)
                draw.text(
                    (box["xyxy"][0], max(0, box["xyxy"][1] - score_font.size - 2)),
                    f"{box['confidence']:.2f}",
                    font=score_font,
                    fill=color,
                    stroke_width=1,
                    stroke_fill="black",
                )
            detail = (
                f"TP={evaluation['tp']} FP={evaluation['fp']} FN={evaluation['fn']} "
                f"| P={percent(evaluation['precision'])} R={percent(evaluation['recall'])}"
            )
            legend = "Orange: matched | red: unmatched | gray: ignored | score: confidence"
        left = column * (width + gap)
        canvas.paste(panel, (left, header))
        titles = ImageDraw.Draw(canvas)
        for row, text in enumerate((headings[column], detail, legend)):
            titles.text((left + 8, 5 + row * (font_size + 9)), text, font=font, fill="white")
    canvas.save(destination)
