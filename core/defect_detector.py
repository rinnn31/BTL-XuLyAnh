import cv2
import numpy as np
from .image_processor import difference_map
from .defect_classifier import classify_region


def detect_defects(template, test, threshold=28, min_area=8, max_area=12000,
                   use_alignment=True, morphology_size=3):
    diff, aligned = difference_map(template, test, use_alignment)
    _, binary = cv2.threshold(diff, int(threshold), 255, cv2.THRESH_BINARY)

    # Loại bỏ nhiễu nhỏ và nối các điểm khác biệt thuộc cùng một vùng lỗi.
    kernel_size = max(1, int(morphology_size))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    edges = cv2.Canny(diff, 35, 110)
    binary = cv2.bitwise_or(binary, cv2.bitwise_and(edges, binary))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    detections = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        label = classify_region(gray, diff, x, y, w, h)
        detections.append({
            "x1": int(x), "y1": int(y), "x2": int(x+w), "y2": int(y+h),
            "width": int(w), "height": int(h), "area_px": area,
            "label": label, "confidence": None
        })

    return {
        "detections": sorted(detections, key=lambda d: d["area_px"], reverse=True),
        "aligned": aligned,
        "difference": diff,
        "mask": binary
    }


def draw_detections(image, detections):
    result = image.copy()
    for index, item in enumerate(detections, 1):
        x1, y1, x2, y2 = item["x1"], item["y1"], item["x2"], item["y2"]
        cv2.rectangle(result, (x1, y1), (x2, y2), (40, 70, 235), 2)
        caption = f'{index}. {item["label"]} | {item["width"]}x{item["height"]}'
        cv2.putText(result, caption, (x1, max(18, y1-7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 70, 235), 1, cv2.LINE_AA)
    return result
