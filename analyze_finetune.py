"""Analyze cached pretrained/best predictions and curves without running a model."""

import argparse
import csv
import json
from pathlib import Path

from comparison_utils import count_metrics, size_group, union_coverage
from data_utils import sha256
from model_arch import ROOT


def analyze(run, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image, ImageDraw

    if output.exists():
        raise FileExistsError("Preserve existing analysis; choose a fresh output")
    result = json.loads((run / "run.json").read_text())
    if result["status"] != "completed" or result["epochs_completed"] != 50 or result["smoke"]:
        raise ValueError("Analyze only the completed formal50epoch run")
    dataset = ROOT / result["config"]["dataset"]
    before_path, after_path = (
        run / "pretrained/predictions.jsonl",
        run / "best_evaluation/predictions.jsonl",
    )
    before = [json.loads(line) for line in before_path.read_text().splitlines()]
    after = [json.loads(line) for line in after_path.read_text().splitlines()]
    if [r["image"] for r in before] != [r["image"] for r in after] or len(before) != 547:
        raise ValueError("Prediction/image membership mismatch")
    output.mkdir(parents=True)
    groups = {}
    per_image = []
    details = {}
    totals = {"recovered_cars": 0, "newly_missed_cars": 0, "new_fp_boxes": 0, "removed_fp_boxes": 0}

    def ious(left, right):
        a = np.asarray(left, dtype=float).reshape(-1, 4)
        b = np.asarray(right, dtype=float).reshape(-1, 4)
        lo = np.maximum(a[:, None, :2], b[None, :, :2])
        hi = np.minimum(a[:, None, 2:], b[None, :, 2:])
        intersection = np.maximum(hi - lo, 0).prod(2)
        area_a = np.maximum(a[:, 2:] - a[:, :2], 0).prod(1)
        area_b = np.maximum(b[:, 2:] - b[:, :2], 0).prod(1)
        return intersection / np.maximum(area_a[:, None] + area_b[None, :] - intersection, 1e-12)

    def match(cars, predictions, regions):
        gt = [list(r[:2]) + [r[0] + r[2], r[1] + r[3]] for r in cars]
        p = sorted([r for r in predictions if r[4] >= 0.25], key=lambda r: -r[4])
        gt_state = ["ignored" if union_coverage(b, regions) >= 0.5 else "fn" for b in gt]
        pred_state = [
            "ignored"
            if r[2] <= r[0] or r[3] <= r[1] or union_coverage(r[:4], regions) >= 0.5
            else "fp"
            for r in p
        ]
        overlap = ious([r[:4] for r in p], gt)
        for index, row in enumerate(overlap):
            if pred_state[index] == "ignored":
                continue
            candidates = [i for i, value in enumerate(row) if value >= 0.5 and gt_state[i] == "fn"]
            if candidates:
                target = min(candidates, key=lambda i: (-row[i], i))
                gt_state[target] = "tp"
                pred_state[index] = "tp"
        counts = count_metrics(pred_state.count("tp"), pred_state.count("fp"), gt_state.count("fn"))
        return {
            "gt": gt,
            "predictions": p,
            "gt_states": gt_state,
            "pred_states": pred_state,
            **counts,
        }

    for old, new in zip(before, after):
        image = old["image"]
        sidecar = json.loads(
            (dataset / "sidecars/val" / Path(image).with_suffix(".json")).read_text()
        )
        regions = [[r[0], r[1], r[0] + r[2], r[1] + r[3]] for r in sidecar["ignored"]]
        a = match(sidecar["cars"], old["predictions"], regions)
        b = match(sidecar["cars"], new["predictions"], regions)
        recovered = sum(x == "fn" and y == "tp" for x, y in zip(a["gt_states"], b["gt_states"]))
        missed = sum(x == "tp" and y == "fn" for x, y in zip(a["gt_states"], b["gt_states"]))
        for index, row in enumerate(sidecar["cars"]):
            if a["gt_states"][index] == "ignored":
                continue
            assert b["gt_states"][index] != "ignored"
            labels = [
                "all",
                "size960_" + size_group(a["gt"][index], sidecar["size"], 960, [16, 32, 96]),
                "occlusion_" + str(row[7]),
            ]
            for label in labels:
                group = groups.setdefault(label, {"gt": 0, "before_tp": 0, "after_tp": 0})
                group["gt"] += 1
                group["before_tp"] += a["gt_states"][index] == "tp"
                group["after_tp"] += b["gt_states"][index] == "tp"
        old_fp = [p[:4] for p, s in zip(a["predictions"], a["pred_states"]) if s == "fp"]
        new_fp = [p[:4] for p, s in zip(b["predictions"], b["pred_states"]) if s == "fp"]
        overlap = ious(new_fp, old_fp)
        used = set()
        paired = 0
        for row in overlap:
            candidates = [i for i, v in enumerate(row) if v >= 0.5 and i not in used]
            if candidates:
                target = min(candidates, key=lambda i: (-row[i], i))
                used.add(target)
                paired += 1
        added, removed = len(new_fp) - paired, len(old_fp) - paired
        values = {
            "recovered_cars": recovered,
            "newly_missed_cars": missed,
            "new_fp_boxes": added,
            "removed_fp_boxes": removed,
        }
        for key, value in values.items():
            totals[key] += value
        per_image.append(
            {
                "image": image,
                "before_tp": a["tp"],
                "after_tp": b["tp"],
                "before_fp": a["fp"],
                "after_fp": b["fp"],
                "before_fn": a["fn"],
                "after_fn": b["fn"],
                **values,
            }
        )
        details[image] = (a, b, sidecar)
    for group in groups.values():
        group.update(
            before_fn=group["gt"] - group["before_tp"],
            after_fn=group["gt"] - group["after_tp"],
            before_recall=group["before_tp"] / group["gt"],
            after_recall=group["after_tp"] / group["gt"],
        )
    for arm, key in [("baseline", "before"), ("best_evaluation", "after")]:
        point = next(
            p
            for p in result[arm]["operating_points"]
            if p["match_iou"] == 0.5 and p["confidence"] == 0.25
        )
        for metric in ("tp", "fp", "fn"):
            if sum(r[f"{key}_{metric}"] for r in per_image) != point[metric]:
                raise ValueError(
                    "Independent error analysis disagrees with saved operating-point counts"
                )
    with (output / "per_image.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)
    summary = {
        "protocol": {
            "confidence": 0.25,
            "match_iou": 0.5,
            "ignore_coverage": 0.5,
            "size_bins": "sqrt bbox area after scaling to960 letterbox: <16,16-32,32-96,>=96",
            "occlusion": "source0=none,1=partial,2=heavy",
            "new_fp_definition": "unmatched to any old FP in confidence-ordered one-to-one IoU>=.50 matching; box changes are not proof of a new physical object",
        },
        "groups": groups,
        "changes": totals,
        "source_sha256": {
            str(p): sha256(p)
            for p in [run / "run.json", run / "results.csv", before_path, after_path]
        },
    }
    (output / "error_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (run / "results.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    rows = [{k.strip(): float(v) for k, v in row.items()} for row in rows]
    epochs = [r["epoch"] for r in rows]
    best_epoch = result["checkpoints"]["best"]["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for loss in ("box_loss", "cls_loss", "dfl_loss"):
        axes[0].plot(epochs, [r["train/" + loss] for r in rows], label=loss)
        axes[1].plot(epochs, [r["val/" + loss] for r in rows], label=loss)
    for metric in ("mAP50(B)", "mAP50-95(B)"):
        axes[2].plot(epochs, [r["metrics/" + metric] for r in rows], label=metric)
    for axis, title in zip(axes, ["Training losses", "Validation losses", "Custom validation AP"]):
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.axvline(best_epoch, color="gray", ls="--", label=f"Best epoch {best_epoch}")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "training_curves.png", dpi=160)
    plt.close(fig)
    selections = [
        ("most_recovered", max(per_image, key=lambda r: r["recovered_cars"])),
        ("new_false_positives", max(per_image, key=lambda r: r["new_fp_boxes"])),
        ("remaining_misses", max(per_image, key=lambda r: r["after_fn"])),
    ]
    for label, row in selections:
        a, b, sidecar = details[row["image"]]
        original = Image.open(dataset / "images/val" / row["image"]).convert("RGB")
        width = min(1100, original.width)
        height = round(original.height * width / original.width)
        canvas = Image.new("RGB", (width * 3, height + 60), "#101820")
        for column, arm in enumerate([None, a, b]):
            picture = original.copy()
            draw = ImageDraw.Draw(picture)
            if arm is None:
                for car in sidecar["cars"]:
                    x, y, w, h = car[:4]
                    draw.rectangle((x, y, x + w, y + h), outline="#00ff66", width=2)
                for ignored in sidecar["ignored"]:
                    x, y, w, h = ignored[:4]
                    draw.rectangle((x, y, x + w, y + h), outline="#aaaaaa", width=3)
                title = "GT: green car, gray ignore"
            else:
                for pred, status in zip(arm["predictions"], arm["pred_states"]):
                    draw.rectangle(
                        pred[:4],
                        outline={"tp": "#ffb000", "fp": "#ff3030", "ignored": "#aaaaaa"}[status],
                        width=2,
                    )
                title = f"{'Pretrained' if column == 1 else 'Fine-tuned'} TP={arm['tp']} FP={arm['fp']} FN={arm['fn']}"
            picture = picture.resize((width, height))
            canvas.paste(picture, (column * width, 60))
            ImageDraw.Draw(canvas).text(
                (column * width + 8, 8),
                title + "\nconf=.25 matchIoU=.50 | orange TP, red FP",
                fill="white",
            )
        canvas.save(output / (label + ".png"))
    lines = [
        "# Batch8 微调结果与错误分析",
        "",
        f"完成50轮；最佳模型来自第{best_epoch}轮。自定义清洗验证集547张，非官方VisDrone AP。",
        "",
        "| 指标 | 预训练 | 微调最佳 |",
        "| --- | ---: | ---: |",
    ]
    for key in ("custom_AP50", "custom_AP75", "custom_AP50_95"):
        lines.append(
            f"| {key} | {result['baseline'][key]:.2%} | {result['best_evaluation'][key]:.2%} |"
        )
    lines += [
        "",
        "| 匹配IoU | Confidence | 原P | 微调P | 原R | 微调R |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]

    def fmt(value):
        return "—" if value is None else f"{value:.2%}"

    for a, b in zip(
        result["baseline"]["operating_points"], result["best_evaluation"]["operating_points"]
    ):
        assert (a["match_iou"], a["confidence"]) == (b["match_iou"], b["confidence"])
        lines.append(
            f"| {a['match_iou']} | {a['confidence']} | {fmt(a['precision'])} | {fmt(b['precision'])} | {fmt(a['recall'])} | {fmt(b['recall'])} |"
        )
    lines += [
        "",
        "以下分组使用conf=.25、匹配IoU=.50；尺寸按缩放到960后的等效框边长划分。",
        "",
        "| 分组 | GT | 原漏检 | 微调漏检 | 原Recall | 微调Recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, g in sorted(groups.items()):
        lines.append(
            f"| {name} | {g['gt']} | {g['before_fn']} | {g['after_fn']} | {g['before_recall']:.2%} | {g['after_recall']:.2%} |"
        )
    lines += [
        "",
        f"原漏检被找回 {totals['recovered_cars']} 辆；原检测成功但微调后漏检 {totals['newly_missed_cars']} 辆。",
        f"按框间IoU≥.50的一对一匹配定义，新出现误检框 {totals['new_fp_boxes']}，消失误检框 {totals['removed_fp_boxes']}。框变化不等于新增物理对象。",
        "",
        f"达到1000框上限的图：预训练 {result['baseline']['capped_images']}，微调 {result['best_evaluation']['capped_images']}。",
        "",
        "![训练验证曲线](training_curves.png)",
        "",
        "[找回最多车辆](most_recovered.png) · [新增误检示例](new_false_positives.png) · [仍有较多漏检](remaining_misses.png)",
        "",
        "单随机种子；最佳轮次在验证集上选择。此结果不是独立测试集泛化结论，短时batch基准也不用于判断检测效果。",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "completed", "output": str(output), "changes": totals}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run.resolve(), args.output.resolve())
