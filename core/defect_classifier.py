import cv2
import numpy as np


def classify_region(gray, diff, x, y, w, h):
    """Ước lượng nhóm lỗi từ hình học và cường độ vùng khác biệt; không dùng mô hình học."""
    roi = diff[y:y+h, x:x+w]
    if roi.size == 0:
        return "Unknown"

    area = max(w * h, 1)
    aspect = max(w, h) / max(min(w, h), 1)
    fill = cv2.countNonZero(cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]) / area
    mean_diff = float(np.mean(roi))

    if aspect > 4.0:
        return "Open" if w >= h else "Short"
    if fill < 0.22 and area < 500:
        return "Pin-hole"
    if aspect > 2.2 and fill < 0.55:
        return "Spur"
    if area < 900 and mean_diff > 45:
        return "Mousebite"
    return "Copper"
