def box_iou(a, b):
    x_left = max(a["x1"], b["x1"])
    y_top = max(a["y1"], b["y1"])
    x_right = min(a["x2"], b["x2"])
    y_bottom = min(a["y2"], b["y2"])
    intersection = max(0, x_right-x_left) * max(0, y_bottom-y_top)
    area_a = max(0, a["x2"]-a["x1"]) * max(0, a["y2"]-a["y1"])
    area_b = max(0, b["x2"]-b["x1"]) * max(0, b["y2"]-b["y1"])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def match_detections(predictions, ground_truth, iou_threshold=0.33, class_aware=False):
    """Ghép dự đoán với nhãn thật theo IoU, mỗi nhãn chỉ được ghép một lần."""
    candidates = []
    for pi, pred in enumerate(predictions):
        for gi, gt in enumerate(ground_truth):
            if class_aware and pred.get("class_id") != gt.get("class_id"):
                continue
            score = box_iou(pred, gt)
            if score >= iou_threshold:
                candidates.append((score, pi, gi))

    used_pred, used_gt = set(), set()
    for score, pi, gi in sorted(candidates, reverse=True):
        if pi not in used_pred and gi not in used_gt:
            used_pred.add(pi)
            used_gt.add(gi)

    tp = len(used_pred)
    fp = len(predictions) - tp
    fn = len(ground_truth) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision,
            "recall": recall, "f1": f1}
