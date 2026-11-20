import cv2
import numpy as np
from . import cv_impl as cv

def read_image(path):
    image = cv2.imread(str(path), cv.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh: {path}")
    return image

def preprocess(image):
    gray = cv.toGray(image)
    return cv.GaussianBlur(gray, (3, 3), 0)


def to_copper_mask(image, white_is_copper=False):
    if image.ndim == 3:
        gray = cv.toGray(image)
    else:
        gray = image

    if white_is_copper:
        return (gray > 127).astype(np.uint8) * 255

    return (gray < 127).astype(np.uint8) * 255

def remove_thin_strips(mask, max_thin=5, min_len=8, min_aspect=5.0):
    """Bỏ các vệt mỏng và dài (lệch biên), giữ lại blob ngắn hoặc dày."""
    n, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        comp = (labels == i).astype(np.uint8)
        pts = cv.findNonZero(comp)
        (_, _), (rw, rh), _ = cv.minAreaRect(pts)
        short, long_ = min(rw, rh) + 1, max(rw, rh) + 1

        
        is_strip =  (short <= max_thin and long_ >= min_len
                    and long_ / short >= min_aspect)
        if not is_strip:
            out[labels == i] = 255
    return out


def _shrink_tol(a, b, tol):
    """Pixel của a mà b không có trong bán kính tol."""
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    return cv.bitwise_and(a, cv.bitwise_not(cv.dilate(b, k)))


def copper_masks(template, test, tol):
    ref_cu = to_copper_mask(template, white_is_copper=False)
    tst_cu = to_copper_mask(test, white_is_copper=False)

    missing = remove_thin_strips(_shrink_tol(ref_cu, tst_cu, tol))
    extra = remove_thin_strips(_shrink_tol(tst_cu, ref_cu, tol))

    return ref_cu, tst_cu, missing, extra