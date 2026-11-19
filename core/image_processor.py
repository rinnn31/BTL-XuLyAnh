import cv2
import numpy as np


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh: {path}")
    return image


def align_to_template(template, test):
    """Căn chỉnh ảnh kiểm tra theo ảnh mẫu bằng ECC khi hai ảnh bị lệch nhẹ."""
    if template.shape[:2] != test.shape[:2]:
        test = cv2.resize(test, (template.shape[1], template.shape[0]))

    ref = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    mov = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    warp = np.eye(2, 3, dtype=np.float32)

    try:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5)
        cv2.findTransformECC(ref, mov, warp, cv2.MOTION_TRANSLATION, criteria)
        aligned = cv2.warpAffine(
            test, warp, (template.shape[1], template.shape[0]),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE
        )
        return aligned
    except cv2.error:
        return test


def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (3, 3), 0)


def to_copper_mask(image, white_is_copper=False):
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    if white_is_copper:
        return (gray > 127).astype(np.uint8) * 255

    return (gray < 127).astype(np.uint8) * 255

def remove_thin_strips(mask, max_thin=5, min_len=8, min_aspect=5.0):
    """Bỏ các vệt mỏng và dài (lệch biên), giữ lại blob ngắn hoặc dày."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        comp = (labels == i).astype(np.uint8)
        pts = cv2.findNonZero(comp)
        (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
        short, long_ = min(rw, rh) + 1, max(rw, rh) + 1

        
        is_strip =  (short <= max_thin and long_ >= min_len
                    and long_ / short >= min_aspect)
        if not is_strip:
            out[labels == i] = 255
    return out


def _shrink_tol(a, b, tol):
    """Pixel của a mà b không có trong bán kính tol."""
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    return cv2.bitwise_and(a, cv2.bitwise_not(cv2.dilate(b, k)))


def _hysteresis(strong, weak):
    """Chỉ giữ các vùng của weak có chứa pixel của strong."""
    _, labels = cv2.connectedComponents(weak, connectivity=8)
    keep = np.unique(labels[strong > 0])
    keep = keep[keep > 0]
    return np.isin(labels, keep).astype(np.uint8) * 255

def copper_masks(template, test, tol_weak=1, tol_strong=1):
    ref_cu = to_copper_mask(template, white_is_copper=False)
    tst_cu = to_copper_mask(test, white_is_copper=False)

    missing = _hysteresis(
        _shrink_tol(ref_cu, tst_cu, tol_strong),
        _shrink_tol(ref_cu, tst_cu, tol_weak)
    )

    extra = _hysteresis(
        _shrink_tol(tst_cu, ref_cu, tol_strong),
        _shrink_tol(tst_cu, ref_cu, tol_weak)
    )

    missing = remove_thin_strips(missing)
    extra = remove_thin_strips(extra)   

    return ref_cu, tst_cu, missing, extra