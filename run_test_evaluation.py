"""Frozen-model full test-dev inference and GT/pretrained/fine-tuned triptychs."""

import argparse
import csv
import html
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from PIL import Image, ImageDraw, ImageFont

from comparison_utils import box_area, diagnostic_match
from data_utils import sha256
from model_arch import ROOT

DEFAULT_CONFIG = ROOT / "configs/test_dev_960.json"


def read_records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def selected_test_records(dataset, count=1610):
    records = [
        r
        for r in read_records(dataset / "manifest.jsonl")
        if r["split"] == "test-dev" and r["included"]
    ]
    declared = (dataset / "test-dev.txt").read_text().splitlines()
    expected = ["./" + r["derived_image"] for r in records]
    if len(records) != count or declared != expected or len(set(expected)) != count:
        raise ValueError("Test-dev membership/order/count differs from frozen manifest")
    for record in records:
        if record["derived_image"] != "images/" + record["id"] + ".jpg":
            raise ValueError("Unexpected image path in test-dev manifest")
    return records


def test_dataset_document(dataset):
    # JSON is valid YAML; no extra parsing dependency needed on Mac.
    return {
        "path": str(dataset),
        "train": str(dataset / "train.txt"),
        "val": str(dataset / "val.txt"),
        "test": str(dataset / "test-dev.txt"),
        "names": ["car"],
        "requires_ignore_aware_trainer": True,
    }


def match_display(sidecar, prediction_rows, confidence=0.25, match_iou=0.50):
    def convert(row):
        x, y, w, h, score, category, truncation, occlusion = row
        return {
            "xyxy": [x, y, x + w, y + h],
            "valid_geometry": w > 0 and h > 0,
            "category_id": category,
            "score": score,
            "truncation": truncation,
            "occlusion": occlusion,
        }

    cars = [convert(row) for row in sidecar["cars"]]
    ignored = [convert(row) for row in sidecar["ignored"]]
    # Degenerate clipped predictions are not evaluable boxes; they were excluded by the evaluator too.
    predictions = [
        {"xyxy": row[:4], "confidence": row[4]}
        for row in prediction_rows
        if row[4] >= confidence and box_area(row[:4]) > 0
    ]
    return cars, ignored, predictions, diagnostic_match(cars, predictions, ignored, match_iou, 0.50)


def draw_triptych(image_path, before, after, target, filename):
    font = ImageFont.load_default(size=20)
    original = Image.open(image_path).convert("RGB")
    width = min(original.width, 1280)
    scale = width / original.width
    height = round(original.height * scale)
    original = original.resize((width, height))
    canvas = Image.new("RGB", (width * 3, height + 124), "#111820")
    colors = {"tp": "#ffb000", "fp": "#ff3030", "ignored": "#aaaaaa"}
    for column, arm in enumerate([None, before, after]):
        panel = original.copy()
        draw = ImageDraw.Draw(panel)
        if arm is None:
            cars, ignored, _, matched = before
            items = [(item["xyxy"], "#aaaaaa") for item in ignored if item["valid_geometry"]]
            items += [(item["xyxy"], "#00ff66") for item in cars]
            title = f"GT car: {len(cars)} | evaluated: {matched['evaluated_gt']}"
            legend = "GREEN: car annotation | GRAY: ignored region"
        else:
            _, _, predictions, matched = arm
            items = [
                (p["xyxy"], colors[s["status"]])
                for p, s in zip(predictions, matched["prediction_states"])
            ]
            title = f"{'Pretrained 960' if column == 1 else 'Fine-tuned 960 (epoch20)'} | TP {matched['tp']} FP {matched['fp']} FN {matched['fn']}"
            legend = "ORANGE: TP | RED: FP | GRAY: ignored"
        for box, color in items:
            draw.rectangle([value * scale for value in box], outline=color, width=2)
        canvas.paste(panel, (column * width, 124))
        ImageDraw.Draw(canvas).multiline_text(
            (column * width + 10, 8),
            f"{title}\n{legend}\nconf >= .25 | match IoU >= .50\n{filename}",
            font=font,
            fill="white",
            spacing=5,
        )
    canvas.save(target, quality=92, subsampling=0)


def gallery(output, rows, featured):
    cards = []
    for row in rows:
        link = "comparisons/" + quote(row["comparison"])
        cards.append(
            f'<article><a href="{link}"><img loading="lazy" src="{link}" alt="{html.escape(row["image"])}"></a>'
            f"<p>{row['index']:04d} · {html.escape(row['image'])}<br>"
            f"漏检 {row['before_fn']} → {row['after_fn']}；误检 {row['before_fp']} → {row['after_fp']}</p></article>"
        )
    shortcuts = " · ".join(
        f'<a href="comparisons/{quote(rows[i]["comparison"])}">{i + 1:04d}</a>' for i in featured
    )
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Test-dev · 960微调前后对比</title><style>
body{font:16px/1.6 system-ui,sans-serif;margin:24px;background:#101820;color:#e7edf4}a{color:#91c9ff}
header{max-width:1100px;margin-bottom:28px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(500px,1fr));gap:20px}
article{border:1px solid #34495e;padding:10px;border-radius:8px}img{width:100%;height:auto}p{margin:8px 0}@media(max-width:550px){.grid{grid-template-columns:1fr}}
</style><header><h1>Test-dev：真实标注｜960微调前｜960微调后</h1>
<p>全部1610张，按冻结文件顺序排列。点击图片放大。绿=car标注，橙=正确检测，红=误检，灰=忽略。
展示阈值conf≥0.25、匹配IoU≥0.50；已固定epoch20模型，不在测试集上调参。</p>
<p>均匀抽取的16张入口（选择仅按序号，不按结果）：SHORTCUTS</p>
<p><a href="report.md">指标报告</a> · <a href="per_image.csv">逐图统计</a></p></header><main class="grid">CARDS</main></html>"""
    (output / "index.html").write_text(
        page.replace("SHORTCUTS", shortcuts).replace("CARDS", "\n".join(cards))
    )


def render_results(dataset, samples, output, config, summaries):
    before = read_records(output / "pretrained/predictions.jsonl")
    after = read_records(output / "finetuned/predictions.jsonl")
    names = [Path(s["derived_image"]).name for s in samples]
    if [r["image"] for r in before] != names or [r["image"] for r in after] != names:
        raise ValueError("Both prediction arms must exactly match frozen test-dev image order")
    images = output / "comparisons"
    images.mkdir()
    rows = []
    for index, (sample, old, new) in enumerate(zip(samples, before, after), 1):
        sidecar = json.loads((dataset / sample["sidecar"]).read_text())
        a = match_display(
            sidecar, old["predictions"], config["display_confidence"], config["display_match_iou"]
        )
        b = match_display(
            sidecar, new["predictions"], config["display_confidence"], config["display_match_iou"]
        )
        target = f"{index:04d}_{Path(old['image']).stem}.jpg"
        draw_triptych(dataset / sample["derived_image"], a, b, images / target, old["image"])
        rows.append(
            {
                "index": index,
                "image": old["image"],
                "comparison": target,
                "raw_gt": len(sidecar["cars"]),
                "evaluated_gt": a[3]["evaluated_gt"],
                **{"before_" + k: a[3][k] for k in ["tp", "fp", "fn"]},
                **{"after_" + k: b[3][k] for k in ["tp", "fp", "fn"]},
            }
        )
        if index % 100 == 0 or index == len(samples):
            print(f"Rendered {index}/{len(samples)} triptychs", flush=True)
    for arm, prefix in [("pretrained", "before"), ("finetuned", "after")]:
        point = next(
            p
            for p in summaries[arm]["operating_points"]
            if p["match_iou"] == 0.5 and p["confidence"] == 0.25
        )
        for key in ("tp", "fp", "fn"):
            if sum(row[prefix + "_" + key] for row in rows) != point[key]:
                raise ValueError("Rendered diagnostic counts disagree with saved test metrics")
    with (output / "per_image.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    featured = [round(i * (len(rows) - 1) / 15) for i in range(16)]
    gallery(output, rows, featured)
    return rows, featured


def run(config, output):
    if os.uname().sysname != "Linux":
        raise RuntimeError("Run inference on MyGPU, not Mac")
    import torch

    from finetune_runtime import write_json
    from model_arch import build_detector, load_config
    from run_finetune import evaluate, verify_prepared

    if output.exists():
        raise FileExistsError("Use a fresh output; preserve existing test evidence")
    if (config["source_split"], config["validator_split"], config["expected_images"]) != (
        "test-dev",
        "test",
        1610,
    ):
        raise ValueError("Only the full frozen1610image test-dev protocol is allowed")
    if (config["imgsz"], config["nms_iou"], config["max_det"], config["eval_conf"]) != (
        960,
        0.70,
        1000,
        0.001,
    ):
        raise ValueError("Keep the previously fixed evaluation protocol")
    if (
        config["display_confidence"],
        config["display_match_iou"],
        config["ignore_coverage"],
        config["selected_epoch"],
    ) != (0.25, 0.50, 0.50, 20):
        raise ValueError("Use fixed display, ignore and epoch-selection settings")
    weights = {
        "pretrained": ROOT / config["initial_weights"],
        "finetuned": ROOT / config["finetuned_weights"],
    }
    expected = {
        "pretrained": config["initial_weights_sha256"],
        "finetuned": config["finetuned_weights_sha256"],
    }
    if any(sha256(weights[name]) != expected[name] for name in weights):
        raise ValueError("Frozen checkpoint hash mismatch")
    dataset, spec = verify_prepared(config)
    samples = selected_test_records(dataset)
    output.mkdir(parents=True)
    yaml = output / "test_data.yaml"
    yaml.write_text(json.dumps(test_dataset_document(dataset), indent=2) + "\n")
    write_json(
        output / "test_manifest.json",
        {
            "split": "test-dev",
            "samples": samples,
            "source_manifest_sha256": spec["manifest_sha256"],
        },
    )
    metadata = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "split": "test-dev",
        "config": config,
        "checkpoint_sha256": expected,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "test_manifest_sha256": sha256(output / "test_manifest.json"),
        "tuning_on_test": False,
        "selected_epoch": 20,
    }
    write_json(output / "run.json", metadata)
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = (
        "1"  # Hash-verified official and own checkpoint only.
    )
    detector, environment = build_detector(load_config(), weights["pretrained"])
    summaries = {}
    summaries["pretrained"] = evaluate(
        detector.model, yaml, output / "pretrained", config, split="test"
    )
    del detector
    torch.cuda.empty_cache()
    summaries["finetuned"] = evaluate(
        str(weights["finetuned"]), yaml, output / "finetuned", config, split="test"
    )
    if any(summary["images"] != 1610 for summary in summaries.values()):
        raise ValueError("Incomplete test inference")
    rows, featured = render_results(dataset, samples, output, config, summaries)
    if any(sha256(weights[name]) != expected[name] for name in weights):
        raise ValueError("Checkpoint bytes changed")
    lines = [
        "# 冻结模型的test-dev960对比",
        "",
        "完整1610张测试图。模型固定为原始预训练权重和昨天验证集选定的第20轮best；没有训练或按测试结果调参。",
        "设置：整图960、FP32、NMS0.70、cap1000、预测保存下限0.001；沿用自定义忽略区域和AP101匹配规则，非官方VisDrone AP。",
        "",
        "| 指标 | 微调前 | 微调后 |",
        "| --- | ---: | ---: |",
    ]
    for key in ["custom_AP50", "custom_AP75", "custom_AP50_95"]:
        lines.append(
            f"| {key} | {summaries['pretrained'][key]:.2%} | {summaries['finetuned'][key]:.2%} |"
        )
    lines += [
        "",
        "| 匹配IoU | Confidence | 原P | 微调P | 原R | 微调R |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for a, b in zip(
        summaries["pretrained"]["operating_points"], summaries["finetuned"]["operating_points"]
    ):
        values = [
            "—" if v is None else f"{v:.2%}"
            for v in [a["precision"], b["precision"], a["recall"], b["recall"]]
        ]
        lines.append(f"| {a['match_iou']} | {a['confidence']} | " + " | ".join(values) + " |")
    lines += [
        "",
        f"可评估GT：{summaries['pretrained']['evaluated_gt']}；忽略GT：{summaries['pretrained']['ignored_gt']}。",
        f"命中1000框上限的图：原模型{summaries['pretrained']['capped_images']}，微调模型{summaries['finetuned']['capped_images']}。",
        "",
        "[全部对比图浏览页](index.html) · [逐图统计](per_image.csv)",
        "",
        "图像左：人工GT（绿car/灰忽略）；中：微调前960；右：微调后960。预测橙TP、红FP、灰忽略。显示confidence≥.25、匹配IoU≥.50。GT列保留原标注，并注明可评估数量。",
        "",
        "以下16张按固定序号均匀取样，不根据效果挑选：",
        "",
    ]
    lines += [
        f"- [{i + 1:04d} {rows[i]['image']}](comparisons/{rows[i]['comparison']})" for i in featured
    ]
    lines += [
        "",
        "这是冻结模型在该公开test-dev上的表现，不保证DJI实拍或任意新场景效果。此后如果根据这次结果调参，这份测试集就参与了开发，不能再当作完全未见的最终验证。",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    metadata.update(
        status="completed",
        completed_at=datetime.now(timezone.utc).isoformat(),
        environment=environment,
        images=1610,
        summaries=summaries,
        comparison_images=len(rows),
        featured_indices=[i + 1 for i in featured],
        diagnostic_count_replay="passed",
        checkpoints_unchanged=True,
    )
    write_json(output / "run.json", metadata)
    print(
        json.dumps(
            {
                "status": "completed",
                "output": str(output),
                "images": 1610,
                "pretrained_AP50_95": summaries["pretrained"]["custom_AP50_95"],
                "finetuned_AP50_95": summaries["finetuned"]["custom_AP50_95"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.execute:
        run(config, args.output.resolve())
    else:
        print(
            json.dumps(
                {"mode": "dry_run", "split": "test-dev", "images": 1610, "config": config}, indent=2
            )
        )
