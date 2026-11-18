import argparse
import csv
from pathlib import Path
from evaluation.batch_evaluate import find_pairs
from core.image_processor import read_image
from core.defect_detector import detect_defects
from core.annotation_reader import read_deeppcb_annotation
from core.metrics import match_detections
from config import REPORT_DIR


def compare(dataset, thresholds=(18, 28, 40, 55)):
    rows = []
    pairs = list(find_pairs(dataset))
    for threshold in thresholds:
        totals = {"tp": 0, "fp": 0, "fn": 0}
        for template_path, test_path, ann_path in pairs:
            result = detect_defects(read_image(template_path), read_image(test_path),
                                    threshold=threshold)
            gt = read_deeppcb_annotation(ann_path)
            score = match_detections(result["detections"], gt)
            for key in totals:
                totals[key] += score[key]
        tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
        p = tp/(tp+fp) if tp+fp else 0
        r = tp/(tp+fn) if tp+fn else 0
        f1 = 2*p*r/(p+r) if p+r else 0
        rows.append({"threshold": threshold, **totals, "precision": p, "recall": r, "f1": f1})

    path = REPORT_DIR / "threshold_comparison.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Đã lưu: {path}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    compare(args.dataset)
