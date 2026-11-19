import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from core.image_processor import read_image
from core.defect_detector import detect_defects, draw_detections
from core.annotation_reader import read_deeppcb_annotation
from core.metrics import box_iou
from config import REPORT_DIR, VISUAL_DIR

# DeepPCB: 1-open, 2-short, 3-mousebite, 4-spur, 5-copper, 6-pin-hole
CLASS_NAMES = {
    1: "Open",
    2: "Short",
    3: "Mousebite",
    4: "Spur",
    5: "Copper",
    6: "Pin-hole",
}
LABELS = list(CLASS_NAMES.values())
NONE = "None"                      # bỏ sót (hàng GT) hoặc phát hiện thừa (cột dự đoán)
ALL_LABELS = LABELS + [NONE]
LABEL_INDEX = {name: i for i, name in enumerate(ALL_LABELS)}


def find_pairs(root):
    """Tìm cặp DeepPCB theo hậu tố _temp.jpg và _test.jpg."""
    root = Path(root)

    for test_path in sorted(root.rglob("*_test.jpg")):
        template_path = test_path.with_name(
            test_path.name.replace("_test.jpg", "_temp.jpg")
        )

        if template_path.exists():
            prefix = test_path.name.replace("_test.jpg", "")
            candidates = [
                # cấu trúc gốc DeepPCB: group00041/00041_not/00041000.txt
                test_path.parent.with_name(test_path.parent.name + "_not") / f"{prefix}.txt",
                test_path.with_name(f"{prefix}.txt"),
                test_path.with_name(f"{prefix}_not.txt"),
                test_path.with_suffix(".txt"),
            ]
            ann_path = next((p for p in candidates if p.exists()), None)

            if ann_path is None:
                print(f"Cảnh báo: không thấy file nhãn cho {test_path.name}, bỏ qua.")
                continue

            yield template_path, test_path, ann_path


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    return precision, recall, f1


def match_pairs(predictions, ground_truth, iou_threshold=0.33, mode="iou", pad=0):
    """Ghép 1-1 (tham lam). Trả về list (pred_idx, gt_idx).

    mode="iou":    IoU >= ngưỡng; box dự đoán có thể nở thêm `pad` px mỗi phía.
    mode="center": tâm box dự đoán nằm trong box nhãn, ưu tiên cặp có tâm gần nhau.
                   Phù hợp vì box nhãn DeepPCB rộng hơn blob chênh lệch nhiều.
    """
    candidates = []
    for pi, pred in enumerate(predictions):
        p = {
            "x1": pred["x1"] - pad, "y1": pred["y1"] - pad,
            "x2": pred["x2"] + pad, "y2": pred["y2"] + pad,
        }
        cx, cy = (p["x1"] + p["x2"]) / 2, (p["y1"] + p["y2"]) / 2
        for gi, gt in enumerate(ground_truth):
            if mode == "center":
                if gt["x1"] <= cx <= gt["x2"] and gt["y1"] <= cy <= gt["y2"]:
                    gx, gy = (gt["x1"] + gt["x2"]) / 2, (gt["y1"] + gt["y2"]) / 2
                    candidates.append((-((cx - gx) ** 2 + (cy - gy) ** 2), pi, gi))
            else:
                score = box_iou(p, gt)
                if score >= iou_threshold:
                    candidates.append((score, pi, gi))

    used_pred, used_gt, pairs = set(), set(), []
    for _, pi, gi in sorted(candidates, reverse=True):
        if pi not in used_pred and gi not in used_gt:
            used_pred.add(pi)
            used_gt.add(gi)
            pairs.append((pi, gi))
    return pairs


def evaluate_pairs(pairs, iou=0.33, save_images=False, verbose=True,
                   match_mode="center", pad=0, **detect_params):
    """Chạy detector trên toàn bộ cặp ảnh.

    Trả về dict gồm rows (theo ảnh), tp/fp/fn và ma trận nhầm lẫn `cm`
    kích thước (6 lớp + None) x (6 lớp + None): hàng = nhãn thật, cột = dự đoán.
      - cm[gt][pred]  : khớp bằng IoU, nhãn thật gt, dự đoán pred
      - cm[gt][None]  : bỏ sót (FN)
      - cm[None][pred]: phát hiện thừa (FP)
    """
    n_labels = len(ALL_LABELS)
    cm = np.zeros((n_labels, n_labels), dtype=np.int64)
    totals = {"tp": 0, "fp": 0, "fn": 0}
    rows = []

    for index, (template_path, test_path, ann_path) in enumerate(pairs, 1):
        try:
            template = read_image(template_path)
            test = read_image(test_path)
        except ValueError as exc:
            print(exc)
            continue

        start = time.perf_counter()
        result = detect_defects(template, test, **detect_params)
        elapsed = (time.perf_counter() - start) * 1000

        detections = result["detections"]
        ground_truth = read_deeppcb_annotation(ann_path)
        matched = match_pairs(detections, ground_truth, iou, match_mode, pad)

        used_pred = {pi for pi, _ in matched}
        used_gt = {gi for _, gi in matched}
        cls_correct = 0

        for pi, gi in matched:
            gt_name = CLASS_NAMES.get(ground_truth[gi]["class_id"], NONE)
            pred_name = detections[pi]["label"]
            if pred_name not in LABEL_INDEX:
                pred_name = NONE
            cm[LABEL_INDEX[gt_name], LABEL_INDEX[pred_name]] += 1
            cls_correct += int(gt_name == pred_name)

        for pi, det in enumerate(detections):
            if pi not in used_pred:
                pred_name = det["label"] if det["label"] in LABEL_INDEX else NONE
                cm[LABEL_INDEX[NONE], LABEL_INDEX[pred_name]] += 1

        for gi, gt in enumerate(ground_truth):
            if gi not in used_gt:
                gt_name = CLASS_NAMES.get(gt["class_id"], NONE)
                cm[LABEL_INDEX[gt_name], LABEL_INDEX[NONE]] += 1

        tp = len(matched)
        fp = len(detections) - tp
        fn = len(ground_truth) - tp
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn

        precision, recall, f1 = prf(tp, fp, fn)
        rows.append({
            "image": test_path.name,
            "gt": len(ground_truth),
            "pred": len(detections),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "cls_correct": cls_correct,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "time_ms": elapsed,
        })

        if save_images:
            visual = draw_detections(result["aligned"], detections)
            VISUAL_DIR.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(VISUAL_DIR / f"{test_path.stem}_result.png"), visual)

        if verbose and index % 50 == 0:
            print(f"Đã xử lý {index}/{len(pairs)} ảnh")

    precision, recall, f1 = prf(totals["tp"], totals["fp"], totals["fn"])
    matched_total = int(sum(cm[i, j] for i in range(len(LABELS)) for j in range(len(LABELS))))
    cls_acc = (
        sum(int(cm[i, i]) for i in range(len(LABELS))) / matched_total
        if matched_total else 0.0
    )

    return {
        "rows": rows,
        **totals,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cls_accuracy": cls_acc,
        "cm": cm,
    }


def print_detection_matrix(tp, fp, fn):
    """Ma trận kết quả phát hiện. TN không xác định trong object detection."""
    print("\n========== DETECTION MATRIX ==========")
    print(f"{'':25} {'Dự đoán khớp':>15} {'Không khớp':>15}")
    print(f"{'Ground Truth':25} {tp:>15} {fn:>15}")
    print(f"{'Dự đoán dư':25} {fp:>15} {'N/A':>15}")
    print("TP: phát hiện đúng | FP: phát hiện thừa | FN: bỏ sót")
    print("TN: không áp dụng trực tiếp cho đánh giá object detection")
    print("======================================\n")


def per_class_metrics(cm):
    """Precision/Recall/F1 từng lớp (tính cả FP/FN do không ghép được)."""
    report = []
    for i, name in enumerate(LABELS):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)      # dự đoán là lớp này nhưng sai/thừa
        fn = int(cm[i, :].sum() - tp)      # nhãn thật là lớp này nhưng sai/bỏ sót
        p, r, f1 = prf(tp, fp, fn)
        report.append({
            "class": name, "support": int(cm[i, :].sum()),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": p, "recall": r, "f1": f1,
        })
    return report


def save_confusion_matrix_csv(cm, output_path):
    with Path(output_path).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["GT \\ Pred"] + ALL_LABELS)
        for name, row in zip(ALL_LABELS, cm):
            writer.writerow([name] + [int(v) for v in row])


def save_class_confusion_matrix(cm, output_path, normalize=False,
                                title="PCB Defect - Confusion Matrix"):
    """Vẽ ma trận nhầm lẫn (hàng = nhãn thật, cột = dự đoán, có thêm lớp None)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = cm.astype(float)
    if normalize:
        sums = data.sum(axis=1, keepdims=True)
        data = np.divide(data, sums, out=np.zeros_like(data), where=sums > 0)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(data, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(ALL_LABELS)))
    ax.set_yticks(range(len(ALL_LABELS)))
    ax.set_xticklabels(ALL_LABELS, rotation=45, ha="right")
    ax.set_yticklabels(ALL_LABELS)

    thresh = data.max() / 2 if data.max() > 0 else 0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = f"{data[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if data[i, j] > thresh else "black")

    ax.set_title(title + (" (chuẩn hoá theo hàng)" if normalize else ""))
    ax.set_xlabel("Predicted Label  (None = không phát hiện)")
    ax.set_ylabel("Ground Truth Label  (None = phát hiện thừa)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def export_confusion_outputs(cm, prefix="", title="PCB Defect - Confusion Matrix"):
    """Xuất CSV + PNG (số lượng và chuẩn hoá) + báo cáo từng lớp vào REPORT_DIR."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = REPORT_DIR / f"{prefix}confusion_matrix.csv"
    png_path = REPORT_DIR / f"{prefix}confusion_matrix.png"
    norm_path = REPORT_DIR / f"{prefix}confusion_matrix_normalized.png"
    cls_path = REPORT_DIR / f"{prefix}per_class_metrics.csv"

    save_confusion_matrix_csv(cm, csv_path)
    save_class_confusion_matrix(cm, png_path, normalize=False, title=title)
    save_class_confusion_matrix(cm, norm_path, normalize=True, title=title)

    report = per_class_metrics(cm)
    with cls_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        writer.writeheader()
        writer.writerows(report)

    return csv_path, png_path, norm_path, cls_path, report


def run(
    dataset,
    min_area=8,
    max_area=4000,
    tol_weak=1,
    tol_strong=1,
    merge_size=5,
    iou=0.33,
    save_images=False,
    use_alignment=False,
    match_mode="center",
    pad=0
):
    pairs = list(find_pairs(dataset))
    print(f"Tìm thấy {len(pairs)} cặp ảnh.")
    if not pairs:
        return []

    out = evaluate_pairs(
        pairs,
        iou=iou,
        save_images=save_images,
        match_mode=match_mode,
        pad=pad,
        min_area=min_area,
        max_area=max_area,
        tol_weak=tol_weak,
        tol_strong=tol_strong,
        merge_size=merge_size,
        use_alignment=use_alignment,
    )
    rows, cm = out["rows"], out["cm"]
    tp, fp, fn = out["tp"], out["fp"], out["fn"]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if rows:
        csv_path = REPORT_DIR / "batch_results.csv"
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV: {csv_path}")

    avg_time = sum(r["time_ms"] for r in rows) / len(rows) if rows else 0.0

    summary = (
        f"Images: {len(rows)}\n"
        f"TP: {tp}\n"
        f"FP: {fp}\n"
        f"FN: {fn}\n"
        f"Precision: {out['precision']:.4f}\n"
        f"Recall: {out['recall']:.4f}\n"
        f"F1: {out['f1']:.4f}\n"
        f"Classification accuracy (trên các box khớp): {out['cls_accuracy']:.4f}\n"
        f"Avg time/image (ms): {avg_time:.2f}\n"
        f"Match mode: {match_mode}\n"
        f"Box pad (px): {pad}\n"
        f"IoU threshold (chỉ dùng khi match=iou): {iou}\n"
        f"Min area: {min_area}\n"
        f"Max area: {max_area}\n"
        f"Tolerance weak: {tol_weak}\n"
        f"Tolerance strong: {tol_strong}\n"
        f"Merge size: {merge_size}\n"
        f"Alignment: {use_alignment}\n"
    )

    summary_path = REPORT_DIR / "summary.txt"
    summary_path.write_text(summary, encoding="utf-8")

    print(summary)
    print_detection_matrix(tp, fp, fn)

    cm_csv, cm_png, cm_norm, cls_csv, report = export_confusion_outputs(cm)
    print("========== PER-CLASS ==========")
    for r in report:
        print(
            f"{r['class']:10} support={r['support']:5} "
            f"P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}"
        )
    print(f"Confusion matrix CSV: {cm_csv}")
    print(f"Confusion matrix PNG: {cm_png}")
    print(f"Confusion matrix (chuẩn hoá): {cm_norm}")
    print(f"Per-class metrics: {cls_csv}")
    print(f"Summary: {summary_path}")

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Đánh giá hàng loạt bộ DeepPCB bằng OpenCV"
    )

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--min-area", type=int, default=8)
    parser.add_argument("--max-area", type=int, default=4000)
    parser.add_argument("--tol-weak", type=int, default=1)
    parser.add_argument("--tol-strong", type=int, default=1)
    parser.add_argument("--merge-size", type=int, default=5)
    parser.add_argument("--iou", type=float, default=0.33)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--align", action="store_true")
    parser.add_argument("--match", choices=["center", "iou"], default="center")
    parser.add_argument("--pad", type=int, default=0)

    args = parser.parse_args()

    run(
        dataset=args.dataset,
        min_area=args.min_area,
        max_area=args.max_area,
        tol_weak=args.tol_weak,
        tol_strong=args.tol_strong,
        merge_size=args.merge_size,
        iou=args.iou,
        save_images=args.save_images,
        use_alignment=args.align,
        match_mode=args.match,
        pad=args.pad
    )