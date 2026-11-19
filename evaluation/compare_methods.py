import argparse
import csv

from evaluation.batch_evaluate import (
    find_pairs,
    print_detection_matrix,
    evaluate_pairs,
    export_confusion_outputs,
    per_class_metrics,
    LABELS,
)
from config import REPORT_DIR

# (tol_weak, tol_strong), luôn cần tol_strong >= tol_weak
DEFAULT_CONFIGS = [(1, 1), (1, 2), (1, 3), (1, 4), (0, 2), (0, 3), (2, 3)]


def parse_configs(text):
    """'1,1 1,2 2,3' -> [(1, 1), (1, 2), (2, 3)]"""
    configs = []
    for item in text.split():
        weak, strong = item.split(",")
        configs.append((int(weak), int(strong)))
    return configs


def compare(
    dataset,
    configs=DEFAULT_CONFIGS,
    min_area=8,
    max_area=4000,
    merge_size=5,
    iou=0.33,
    use_alignment=False,
    match_mode="center",
    pad=0
):
    rows = []
    class_rows = []
    cms = {}
    pairs = list(find_pairs(dataset))

    if not pairs:
        print("Không tìm thấy cặp ảnh DeepPCB.")
        return rows

    print(f"Tìm thấy {len(pairs)} cặp ảnh | match={match_mode} pad={pad}")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    for tol_weak, tol_strong in configs:
        if tol_strong < tol_weak:
            print(f"Bỏ qua ({tol_weak},{tol_strong}): tol_strong phải >= tol_weak")
            continue

        name = f"weak{tol_weak}_strong{tol_strong}"

        out = evaluate_pairs(
            pairs,
            iou=iou,
            verbose=False,
            match_mode=match_mode,
            pad=pad,
            min_area=min_area,
            max_area=max_area,
            tol_weak=tol_weak,
            tol_strong=tol_strong,
            merge_size=merge_size,
            use_alignment=use_alignment,
        )

        tp, fp, fn = out["tp"], out["fp"], out["fn"]
        avg_time = (
            sum(r["time_ms"] for r in out["rows"]) / len(out["rows"])
            if out["rows"] else 0.0
        )

        rows.append({
            "config": name,
            "tol_weak": tol_weak,
            "tol_strong": tol_strong,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": out["precision"],
            "recall": out["recall"],
            "f1": out["f1"],
            "cls_accuracy": out["cls_accuracy"],
            "avg_time_ms": avg_time,
        })
        cms[name] = out["cm"]

        for r in per_class_metrics(out["cm"]):
            class_rows.append({"config": name, **r})

        print(f"\n========== {name} ==========")
        print_detection_matrix(tp, fp, fn)
        print(
            f"Precision: {out['precision']:.4f}\n"
            f"Recall:    {out['recall']:.4f}\n"
            f"F1-score:  {out['f1']:.4f}\n"
            f"Class acc: {out['cls_accuracy']:.4f}"
        )

        # Mỗi cấu hình xuất riêng một ma trận nhầm lẫn
        export_confusion_outputs(
            out["cm"],
            prefix=f"{name}_",
            title=f"Confusion Matrix ({name})",
        )

    if not rows:
        return rows

    with (REPORT_DIR / "tolerance_comparison.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with (REPORT_DIR / "tolerance_per_class.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(class_rows[0].keys()))
        writer.writeheader()
        writer.writerows(class_rows)

    # ---- Bảng tổng hợp (sắp theo F1) ----
    print("\n========== SO SÁNH CẤU HÌNH (sắp theo F1) ==========")
    for row in sorted(rows, key=lambda r: r["f1"], reverse=True):
        print(
            f"{row['config']:16} | "
            f"TP={row['tp']:5} FP={row['fp']:5} FN={row['fn']:5} | "
            f"P={row['precision']:.4f} R={row['recall']:.4f} "
            f"F1={row['f1']:.4f} | ClsAcc={row['cls_accuracy']:.4f}"
        )

    # ---- Recall từng lớp: để thấy lớp nhỏ (Spur, Mousebite...) có bị mất không ----
    print("\n========== RECALL THEO LỚP ==========")
    print(f"{'config':16} " + " ".join(f"{c:>10}" for c in LABELS))
    for row in rows:
        recalls = {
            r["class"]: r["recall"]
            for r in class_rows if r["config"] == row["config"]
        }
        print(
            f"{row['config']:16} "
            + " ".join(f"{recalls[c]:>10.3f}" for c in LABELS)
        )

    best = max(rows, key=lambda r: r["f1"])
    print(f"\nCấu hình tốt nhất theo F1: {best['config']} (F1={best['f1']:.4f})")
    export_confusion_outputs(
        cms[best["config"]],
        prefix="best_",
        title=f"Confusion Matrix - best ({best['config']})",
    )

    print(f"\nĐã lưu: {REPORT_DIR / 'tolerance_comparison.csv'}")
    print(f"Đã lưu: {REPORT_DIR / 'tolerance_per_class.csv'}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="So sánh các cấu hình tolerance (weak/strong) của bộ phát hiện lỗi PCB"
    )

    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--configs",
        default=None,
        help='Danh sách "weak,strong" cách nhau bằng khoảng trắng, vd: "1,1 1,2 1,3"',
    )
    parser.add_argument("--min-area", type=int, default=8)
    parser.add_argument("--max-area", type=int, default=4000)
    parser.add_argument("--merge-size", type=int, default=5)
    parser.add_argument("--iou", type=float, default=0.33)
    parser.add_argument("--align", action="store_true")
    parser.add_argument("--match", choices=["center", "iou"], default="center")
    parser.add_argument("--pad", type=int, default=0)

    args = parser.parse_args()

    compare(
        dataset=args.dataset,
        configs=parse_configs(args.configs) if args.configs else DEFAULT_CONFIGS,
        min_area=args.min_area,
        max_area=args.max_area,
        merge_size=args.merge_size,
        iou=args.iou,
        use_alignment=args.align,
        match_mode=args.match,
        pad=args.pad
    )