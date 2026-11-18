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


def difference_map(template, test, use_alignment=True):
    if use_alignment:
        test = align_to_template(template, test)
    ref_gray = preprocess(template)
    test_gray = preprocess(test)
    diff = cv2.absdiff(ref_gray, test_gray)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    return diff, test
