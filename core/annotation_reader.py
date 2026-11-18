from pathlib import Path


def read_deeppcb_annotation(path):
    """Đọc nhãn DeepPCB: x1,y1,x2,y2,class hoặc các trường phân tách bằng khoảng trắng."""
    items = []
    path = Path(path)
    if not path.exists():
        return items

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip().replace(",", " ")
        if not line:
            continue
        values = line.split()
        if len(values) < 5:
            continue
        try:
            x1, y1, x2, y2, class_id = map(float, values[:5])
            items.append({
                "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                "class_id": int(class_id)
            })
        except ValueError:
            continue
    return items
