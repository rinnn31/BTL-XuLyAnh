import numpy as np
from . import cv_impl as cv

# Phân loại theo luật hình học và tô-pô, không dùng mô hình học.
# Thiếu đồng: Open, Mousebite, Pin-hole. Thừa đồng: Short, Spur, Copper.


def _ncomp(mask):
    return cv.connectedComponents(mask, connectivity=8)[0] - 1


def _topology_delta(ref_cu, tst_cu, x, y, w, h, pad=8):
    """Số thành phần liên thông của đồng (test - mẫu) trong vùng quanh lỗi."""
    H, W = ref_cu.shape
    s = (slice(max(y - pad, 0), min(y + h + pad, H)),
         slice(max(x - pad, 0), min(x + w + pad, W)))
    return _ncomp(tst_cu[s]) - _ncomp(ref_cu[s])


def _ring(blob, r):
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv.bitwise_and(cv.dilate(blob, k), cv.bitwise_not(blob))


def _copper_arcs(ref_cu, blob, r=3):
    """Số cung đồng (của ảnh mẫu) nằm trên vòng quanh blob."""
    arcs = cv.bitwise_and(_ring(blob, r), ref_cu)
    n, _, stats, _ = cv.connectedComponentsWithStats(arcs, connectivity=8)
    return sum(1 for i in range(1, n) if stats[i, cv.CC_STAT_AREA] >= 2 * r)


def _ring_copper_ratio(mask, blob, r=3):
    """Tỉ lệ vòng quanh blob nằm trên mask."""
    ring = _ring(blob, r)
    n = cv.countNonZero(ring)
    return cv.countNonZero(cv.bitwise_and(ring, mask)) / n if n else 0.0


def region_sign(ref_g, tst_g, comp, bright):
    """>0: ảnh test có nhiều đồng hơn (thừa). <0: thiếu đồng.

    Dùng chênh lệch độ xám có dấu thay vì đếm pixel mask nên ít phụ thuộc
    ngưỡng nhị phân và độ dày nét vẽ của hai ảnh.
    """
    d = tst_g.astype(np.float32) - ref_g.astype(np.float32)
    d -= np.median(d)              # bù lệch sáng toàn cục giữa hai ảnh
    if not bright:
        d = -d
    vals = d[comp > 0]
    return float(vals.sum()) if vals.size else 0.0


def classify_missing(ref_cu, tst_cu, blob, x, y, w, h):
    if _copper_arcs(ref_cu, blob) >= 2:
        return "Open"              # đồng chỉ còn ở hai phía đối diện: mạch bị cắt
    if _ring_copper_ratio(ref_cu, blob) > 0.9:
        return "Pin-hole"          # đồng bao kín xung quanh
    return "Mousebite"             # đồng chỉ còn một phía: khuyết ở mép


def classify_extra(ref_cu, tst_cu, blob, x, y, w, h):
    if _topology_delta(ref_cu, tst_cu, x, y, w, h) < 0:
        return "Short"             # nối hai mạch riêng biệt
    if _ring_copper_ratio(ref_cu, blob, r=2) > 0.05:
        return "Spur"              # thừa đồng bám vào mạch
    return "Copper"                # mảng đồng thừa tách rời