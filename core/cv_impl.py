import math
import numpy as np

IMREAD_COLOR = 1
COLOR_BGR2GRAY = 6
INTER_LINEAR = 1
WARP_INVERSE_MAP = 16
BORDER_REPLICATE = 1
MOTION_TRANSLATION = 0
TERM_CRITERIA_EPS = 2
TERM_CRITERIA_COUNT = 1
MORPH_ELLIPSE = 2
MORPH_CLOSE = 3
CC_STAT_LEFT, CC_STAT_TOP, CC_STAT_WIDTH, CC_STAT_HEIGHT, CC_STAT_AREA = range(5)
FONT_HERSHEY_SIMPLEX = 0
LINE_AA = 16

# Exception placeholder để tương thích với khối except cv2.error.
error = Exception


def _u8(a):
    return np.clip(np.rint(a), 0, 255).astype(np.uint8)


def toGray(image,):
    a = np.asarray(image)
    if a.ndim == 2:
        return a.copy()
    # Công thức độ sáng tiêu chuẩn cho ảnh BGR.
    return _u8(0.114 * a[..., 0].astype(float) +
               0.587 * a[..., 1].astype(float) +
               0.299 * a[..., 2].astype(float))

def GaussianBlur(image, ksize, sigmaX=0):
    """Gaussian blur separable; sigma tự suy ra nếu sigmaX=0."""
    a = np.asarray(image).astype(float)
    kh, kw = map(int, ksize[1::-1])  # ksize OpenCV là (width, height)
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("Kernel Gaussian phải có kích thước lẻ")
    sigma = float(sigmaX) if sigmaX > 0 else max(0.3 * ((kw - 1) * 0.5 - 1) + 0.8, 0.3)
    def kernel(n):
        x = np.arange(n) - n // 2
        g = np.exp(-(x*x) / (2*sigma*sigma))
        return g / g.sum()
    gx, gy = kernel(kw), kernel(kh)
    def conv_axis(arr, k, axis):
        pad = len(k)//2
        pads = [(0, 0)] * arr.ndim
        pads[axis] = (pad, pad)
        z = np.pad(arr, pads, mode="edge")
        out = np.zeros_like(arr, dtype=float)
        for i, weight in enumerate(k):
            sl = [slice(None)] * arr.ndim
            sl[axis] = slice(i, i + arr.shape[axis])
            out += weight * z[tuple(sl)]
        return out
    out = conv_axis(a, gx, 1)
    out = conv_axis(out, gy, 0)
    return _u8(out)


def _kernel_array(kernel):
    k = np.asarray(kernel)
    if k.ndim != 2:
        raise ValueError("Kernel phải là ma trận 2 chiều")
    return k != 0


def getStructuringElement(shape, ksize):
    w, h = map(int, ksize)
    yy, xx = np.mgrid[:h, :w]
    if shape == MORPH_ELLIPSE:
        # Ellipse nội tiếp hình chữ nhật kernel.
        rx, ry = max((w-1)/2, 0.5), max((h-1)/2, 0.5)
        return (((xx-(w-1)/2)/rx)**2 + ((yy-(h-1)/2)/ry)**2 <= 1).astype(np.uint8)
    return np.ones((h, w), np.uint8)


def _morph(image, kernel, operation):
    a = np.asarray(image) > 0
    k = _kernel_array(kernel)
    kh, kw = k.shape
    py, px = kh//2, kw//2
    padded = np.pad(a, ((py, py), (px, px)), mode="constant",
                    constant_values=(False if operation == "dilate" else True))
    out = np.zeros_like(a) if operation == "dilate" else np.ones_like(a)
    for y in range(kh):
        for x in range(kw):
            if not k[y, x]:
                continue
            part = padded[y:y+a.shape[0], x:x+a.shape[1]]
            if operation == "dilate":
                out |= part
            else:
                out &= part
    return out.astype(np.uint8) * 255


def dilate(image, kernel, iterations=1):
    out = np.asarray(image)
    for _ in range(max(1, int(iterations))):
        out = _morph(out, kernel, "dilate")
    return out


def erode(image, kernel, iterations=1):
    out = np.asarray(image)
    for _ in range(max(1, int(iterations))):
        out = _morph(out, kernel, "erode")
    return out


def morphologyEx(image, op, kernel):
    if op != MORPH_CLOSE:
        raise NotImplementedError("mini_cv chỉ hỗ trợ MORPH_CLOSE")
    return erode(dilate(image, kernel), kernel)


def bitwise_and(a, b):
    return np.bitwise_and(np.asarray(a, dtype=np.uint8), np.asarray(b, dtype=np.uint8))


def bitwise_or(a, b):
    return np.bitwise_or(np.asarray(a, dtype=np.uint8), np.asarray(b, dtype=np.uint8))


def bitwise_not(a):
    return np.bitwise_not(np.asarray(a, dtype=np.uint8))


def countNonZero(a):
    return int(np.count_nonzero(a))


def findNonZero(image):
    ys, xs = np.nonzero(image)
    if len(xs) == 0:
        return None
    return np.column_stack((xs, ys)).astype(np.int32).reshape(-1, 1, 2)


def connectedComponents(image, connectivity=8):
    n, labels, _, _ = connectedComponentsWithStats(image, connectivity)
    return n, labels


def connectedComponentsWithStats(image, connectivity=8):
    """Connected-component labeling bằng flood-fill/BFS, nhãn nền=0."""
    src = np.asarray(image) != 0
    h, w = src.shape
    labels = np.zeros((h, w), dtype=np.int32)
    stats = [[0, 0, w, h, int((~src).sum())]]
    neighbors = [(-1,0),(1,0),(0,-1),(0,1)]
    if connectivity == 8:
        neighbors += [(-1,-1),(-1,1),(1,-1),(1,1)]
    label = 0
    for sy, sx in zip(*np.nonzero(src & (labels == 0))):
        if labels[sy, sx] != 0:
            continue
        label += 1
        stack = [(int(sy), int(sx))]
        labels[sy, sx] = label
        minx = maxx = int(sx)
        miny = maxy = int(sy)
        area = 0
        while stack:
            y, x = stack.pop()
            area += 1
            minx, maxx = min(minx, x), max(maxx, x)
            miny, maxy = min(miny, y), max(maxy, y)
            for dy, dx in neighbors:
                ny, nx = y+dy, x+dx
                if 0 <= ny < h and 0 <= nx < w and src[ny, nx] and labels[ny, nx] == 0:
                    labels[ny, nx] = label
                    stack.append((ny, nx))
        stats.append([minx, miny, maxx-minx+1, maxy-miny+1, area])
    stats = np.asarray(stats, dtype=np.int32)
    centroids = np.zeros((label+1, 2), dtype=float)
    for i in range(label+1):
        yy, xx = np.nonzero(labels == i)
        if len(xx):
            centroids[i] = [xx.mean(), yy.mean()]
    return label+1, labels, stats, centroids


def minAreaRect(points):
    """Xấp xỉ hình chữ nhật xoay bằng PCA; trả ((cx,cy),(w,h),angle)."""
    p = np.asarray(points).reshape(-1, 2).astype(float)
    if len(p) == 0:
        return ((0., 0.), (0., 0.), 0.)
    center = p.mean(axis=0)
    if len(p) == 1:
        return ((float(center[0]), float(center[1])), (0., 0.), 0.)
    cov = np.cov((p-center).T)
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, np.argmax(vals)]
    angle = math.degrees(math.atan2(axis[1], axis[0]))
    rot = np.array([[axis[0], axis[1]], [-axis[1], axis[0]]])
    q = (p-center) @ rot.T
    lo, hi = q.min(axis=0), q.max(axis=0)
    size = hi-lo
    return ((float(center[0]), float(center[1])),
            (float(size[0]), float(size[1])), float(angle))
