# app/utils/visualize.py
from typing import List, Tuple, Optional, Union, Dict, Any
import cv2
import numpy as np

ColorSpec = Union[str, Tuple[int, int, int]]  # "#RRGGBB" или BGR-ту플

# ---------- utils: colors ----------

def hex_to_bgr(hexstr: str) -> Tuple[int, int, int]:
    s = hexstr.strip().lstrip("#")
    if len(s) != 6:
        return (0, 200, 0)
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return (b, g, r)

def _ensure_bgr(c: ColorSpec) -> Tuple[int, int, int]:
    if isinstance(c, tuple) and len(c) == 3:
        # предполагаем BGR
        return (int(c[0]), int(c[1]), int(c[2]))
    if isinstance(c, str):
        return hex_to_bgr(c)
    return (0, 200, 0)

def _auto_palette(i: int) -> Tuple[int, int, int]:
    base = [
        (36, 255, 12),    # зелёный
        (0, 165, 255),    # оранжевый
        (0, 0, 255),      # красный
        (255, 0, 0),      # синий
        (255, 0, 255),    # фиолетовый
        (255, 255, 0),    # жёлто-циан
    ]
    return base[i % len(base)]

def _pick_color(
    idx: int,
    item_cls: Optional[str],
    palette: Optional[Union[List[ColorSpec], Dict[str, ColorSpec]]],
) -> Tuple[int, int, int]:
    if palette is None:
        return _auto_palette(idx)
    if isinstance(palette, dict) and item_cls is not None:
        c = palette.get(item_cls)
        if c is not None:
            return _ensure_bgr(c)
        # если нет точного класса, попробуем 'default'
        c = palette.get("default")
        if c is not None:
            return _ensure_bgr(c)
        return _auto_palette(idx)
    if isinstance(palette, list) and len(palette) > 0:
        return _ensure_bgr(palette[idx % len(palette)])
    # любой другой случай
    return _auto_palette(idx)

# ---------- utils: boxes ----------

def _xyxy_from_bbox(bbox: Any) -> Tuple[int, int, int, int]:
    """
    Приводит bbox к (x1, y1, x2, y2) с int.
    Поддерживает dict {"x1","y1","x2","y2"} и list/tuple [x1,y1,x2,y2].
    """
    if isinstance(bbox, dict):
        x1 = int(bbox.get("x1", 0))
        y1 = int(bbox.get("y1", 0))
        x2 = int(bbox.get("x2", 0))
        y2 = int(bbox.get("y2", 0))
    elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x1, y1, x2, y2 = map(int, bbox)
    else:
        x1 = y1 = x2 = y2 = 0

    # упорядочим координаты на случай, если пришли перепутанные
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2

def _clamp_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2

def _draw_label(img: np.ndarray, x: int, y: int, text: str, color: Tuple[int, int, int]) -> None:
    text = str(text)
    fs = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, fs, scale, thickness)
    pad = 2
    x2 = x + tw + pad * 2
    y2 = y + th + pad * 2
    x = max(0, x); y = max(0, y)
    x2 = min(img.shape[1] - 1, x2)
    y2 = min(img.shape[0] - 1, y2)
    # подложка
    cv2.rectangle(img, (x, y), (x2, y2), (0, 0, 0), -1)
    cv2.putText(img, text, (x + pad, y + th + pad), fs, scale, color, thickness, cv2.LINE_AA)

# ---------- main overlay ----------

def overlay_instances(
    image: np.ndarray,
    plants: List[Dict[str, Any]],
    defects: List[Dict[str, Any]],
    plant_colors: Optional[Union[List[ColorSpec], Dict[str, ColorSpec]]] = None,
    defect_colors: Optional[Union[List[ColorSpec], Dict[str, ColorSpec]]] = None,
) -> np.ndarray:
    """
    Надёжная визуализация:
      - маски накладываются с альфа-блендингом
      - bbox нормализуются в (x1,y1,x2,y2) и кланпятся в границы
      - цвета берутся из dict по имени класса, либо из списка, либо авто-палитра
      - подписи не ломают пайплайн, даже если conf отсутствует
    """
    out = image.copy()
    H, W = out.shape[:2]

    # Растения
    for i, p in enumerate(plants):
        col = _pick_color(i, p.get("cls"), plant_colors)

        # Маска
        m = p.get("mask")
        if isinstance(m, np.ndarray) and m.ndim >= 2:
            m_bin = (m > 0).astype(np.uint8)
            if m_bin.shape[:2] != (H, W):
                m_bin = cv2.resize(m_bin, (W, H), interpolation=cv2.INTER_NEAREST)
            # альфа-блендинг по маске
            alpha = 0.4
            color_arr = np.full_like(out, col, dtype=np.uint8)
            mask3 = m_bin.astype(bool)
            out[mask3] = (out[mask3].astype(np.float32) * (1 - alpha) + color_arr[mask3].astype(np.float32) * alpha).astype(np.uint8)

        # Бокс
        x1, y1, x2, y2 = _xyxy_from_bbox(p.get("bbox"))
        x1, y1, x2, y2 = _clamp_box(x1, y1, x2, y2, W, H)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)

        # Подпись
        cls_name = p.get("species") or p.get("cls") or "plant"
        conf = p.get("conf", None)
        if isinstance(conf, (int, float)):
            label = f"P{int(p.get('id', i+1))}:{cls_name} {float(conf):.2f}"
        else:
            label = f"P{int(p.get('id', i+1))}:{cls_name}"
        _draw_label(out, x1, max(0, y1 - 18), label, col)

    # Дефекты
    for j, d in enumerate(defects):
        col = _pick_color(j, d.get("cls"), defect_colors)

        # Маска
        m = d.get("mask")
        if isinstance(m, np.ndarray) and m.ndim >= 2:
            m_bin = (m > 0).astype(np.uint8)
            if m_bin.shape[:2] != (H, W):
                m_bin = cv2.resize(m_bin, (W, H), interpolation=cv2.INTER_NEAREST)
            alpha = 0.4
            color_arr = np.full_like(out, col, dtype=np.uint8)
            mask3 = m_bin.astype(bool)
            out[mask3] = (out[mask3].astype(np.float32) * (1 - alpha) + color_arr[mask3].astype(np.float32) * alpha).astype(np.uint8)

        # Бокс
        x1, y1, x2, y2 = _xyxy_from_bbox(d.get("bbox"))
        x1, y1, x2, y2 = _clamp_box(x1, y1, x2, y2, W, H)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)

        # Подпись
        cls_name = d.get("cls") or "defect"
        conf = d.get("conf", None)
        base = f"D{int(d.get('id', j+1))}:{cls_name}"
        if isinstance(conf, (int, float)):
            base += f" {float(conf):.2f}"
        pid = d.get("plant_id", None)
        if pid is not None:
            base += f" -> P{int(pid)}"
        sev = d.get("severity", None)
        if isinstance(sev, str):
            base += f" [{sev}]"
        _draw_label(out, x1, min(H - 22, y2 + 4), base, col)

    return out

# ---------- analytics helpers ----------

def compute_tilt_deg(mask: np.ndarray) -> float:
    """
    Оценка наклона основного направления контура относительно вертикали.
    Возвращает [0..90]; 0 — вертикально, 90 — горизонтально.
    """
    ys, xs = np.where(mask > 0)
    if xs.size < 10:
        return 0.0
    pts = np.vstack([xs, ys]).T.astype(np.float32)
    rect = cv2.minAreaRect(pts)  # ((cx,cy),(w,h),angle[-90..0))
    angle = rect[-1]
    if angle < -45:
        angle = angle + 90
    tilt_from_vertical = abs(90 - abs(angle))
    return float(round(tilt_from_vertical, 2))

def compute_dry_ratio(image_bgr: np.ndarray, mask: np.ndarray) -> float:
    """
    Наивная сухость: доля пикселей с низкой насыщенностью
    или в коричнево-жёлтом диапазоне при невысокой яркости.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    m = (mask > 0)
    if m.sum() == 0:
        return 0.0
    dry = ((s < 50) | (((h >= 10) & (h <= 25)) & (v < 180)))[m]
    return float(round(dry.mean(), 3))