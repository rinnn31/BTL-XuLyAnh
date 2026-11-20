import cv2
from . import cv_impl as cv
from .image_processor import copper_masks, preprocess
from .defect_classifier import classify_missing, classify_extra, region_sign


def detect_defects(template, test, min_area=8, max_area=4000, tol = 1, merge_size=5, **_):

    ref_cu, tst_cu, missing, extra = copper_masks(
        template, test, tol)
    ref_g, tst_g = preprocess(template), preprocess(test)

    # Gom missing + extra thành các vùng khác biệt chung
    both = cv.bitwise_or(missing, extra)
    merge = cv.getStructuringElement(cv.MORPH_ELLIPSE, (merge_size, merge_size))
    both = cv.morphologyEx(both, cv.MORPH_CLOSE, merge)
    n, labels, stats, _ = cv.connectedComponentsWithStats(both, connectivity=8)

    detections = []
    for i in range(1, n):
        x, y, w, h, _a = stats[i]
        comp = (labels == i).astype("uint8") * 255
        m_blob = cv.bitwise_and(missing, comp)
        e_blob = cv.bitwise_and(extra, comp)
        m, e = cv.countNonZero(m_blob), cv.countNonZero(e_blob)
        area = m + e
        if area < min_area or area > max_area:
            continue


        # Dấu chênh lệch độ xám quyết định thừa hay thiếu đồng
        if region_sign(ref_g, tst_g, comp, False) < 0:
            blob, classify = (m_blob if m else comp), classify_missing
        else:
            blob, classify = (e_blob if e else comp), classify_extra
        label = classify(ref_cu, tst_cu, blob, int(x), int(y), int(w), int(h))

        detections.append({
            "x1": int(x), "y1": int(y), "x2": int(x + w), "y2": int(y + h),
            "width": int(w), "height": int(h), "area_px": float(area),
            "label": label, "confidence": None
        })

    return {
        "detections": sorted(detections, key=lambda d: d["area_px"], reverse=True),
        "mask": both
    }


def draw_detections(image, detections):
    result = image.copy()
    for index, item in enumerate(detections, 1):
        x1, y1, x2, y2 = item["x1"], item["y1"], item["x2"], item["y2"]
        cv2.rectangle(result, (x1, y1), (x2, y2), (40, 70, 235), 2)
        caption = f'{index}. {item["label"]} | {item["width"]}x{item["height"]}'
        cv2.putText(result, caption, (x1, max(18, y1 - 7)),
                    cv.FONT_HERSHEY_SIMPLEX, 0.42, (40, 70, 235), 1, cv.LINE_AA)
    return result