import argparse
import csv
import time
from pathlib import Path
import cv2

from core.defect_detector import detect_defects, draw_detections
from core.annotation_reader import read_deeppcb_annotation
from core.metrics import match_detections
from config import REPORT_DIR, VISUAL_DIR


def find_pairs(root):
    """Tìm cặp DeepPCB theo hậu tố _temp.jpg và _test.jpg."""
    root = Path(root)
    for test_path in root.rglob("*_test.jpg"):
        template_path = test_path.with_name(test_path.name.replace("_test.jpg", "_temp.jpg"))
        if template_path.exists():
            ann_path = test_path.with_name(test_path.name.replace("_test.jpg", ".txt"))
            if not ann_path.exists():
                ann_path = test_path.with_suffix(".txt")
            yield template_path, test_path, ann_path


def run(dataset, threshold=28, min_area=8, iou=0.33, save_images=False):
    rows = []
    totals = {"tp": 0, "fp": 0, "fn": 0}
    pairs = list(find_pairs(dataset))
    for index, (template_path, test_path, ann_path) in enumerate(pairs, 1):
        template = cv2.imread(str(template_path))
        test = cv2.imread(str(test_path))
        if template is None or test is None:
            continue

        start = time.perf_counter()
        result = detect_defects(template, test, threshold=threshold, min_area=min_area)
        elapsed = (time.perf_counter() - start) * 1000
        ground_truth = read_deeppcb_annotation(ann_path)
        # Đánh giá định vị không phụ thuộc nhãn vì bộ phân loại heuristic không tương đương nhãn học.
        metrics = match_detections(result["detections"], ground_truth, iou, class_aware=False)
        for key in totals:
            totals[key] += metrics[key]

        rows.append({
            "image": test_path.name, "gt": len(ground_truth),
            "pred": len(result["detections"]), "tp": metrics["tp"],
            "fp": metrics["fp"], "fn": metrics["fn"],
            "precision": metrics["precision"], "recall": metrics["recall"],
            "f1": metrics["f1"], "time_ms": elapsed
        })

        if save_images:
            visual = draw_detections(result["aligned"], result["detections"])
            cv2.imwrite(str(VISUAL_DIR / f"{test_path.stem}_result.png"), visual)
        if index % 50 == 0:
            print(f"Đã xử lý {index}/{len(pairs)} ảnh")

    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    precision = tp/(tp+fp) if tp+fp else 0
    recall = tp/(tp+fn) if tp+fn else 0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0

    csv_path = REPORT_DIR / "batch_results.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary_path = REPORT_DIR / "summary.txt"
    summary_path.write_text(
        f"Images: {len(rows)}\nTP: {tp}\nFP: {fp}\nFN: {fn}\\n"
        f"Precision: {precision:.4f}\nRecall: {recall:.4f}\nF1: {f1:.4f}\\n",
        encoding="utf-8"
    )
    print(summary_path.read_text(encoding="utf-8"))
    print(f"CSV: {csv_path}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá hàng loạt bộ DeepPCB bằng OpenCV")
    parser.add_argument("--dataset", required=True, help="Thư mục gốc DeepPCB")
    parser.add_argument("--threshold", type=int, default=28)
    parser.add_argument("--min-area", type=int, default=8)
    parser.add_argument("--iou", type=float, default=0.33)
    parser.add_argument("--save-images", action="store_true")
    args = parser.parse_args()
    run(args.dataset, args.threshold, args.min_area, args.iou, args.save_images)
