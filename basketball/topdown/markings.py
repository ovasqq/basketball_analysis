"""Разметка корта: шаблон FIBA в метрах и подсветка его на кадре через гомографию.

Разметка не ищется на фото, а рисуется по шаблону: линии в метрах корта
проецируются в пиксели через CourtHomography.court_to_image. Поэтому точность
подсветки равна точности калибровки. Перекрытие линий посторонними объектами
(мяч, игроки) пока не учитывается -- см. OCCLUSION.md.

Из кода:
    ch = CourtHomography.from_json("calibration.json", image_size=(w, h))
    cv2.imwrite("frame_lightened.jpg", draw_markings(ch, frame))
"""
from __future__ import annotations

import cv2
import numpy as np


def _arc(cx, cy, r, a0, a1, n=64):
    t = np.linspace(a0, a1, n)
    return np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])


def court_lines(L=28.0, W=15.0):
    """Разметка FIBA в метрах (оси как в CourtHomography): список полилиний (N, 2).
    Размеры элементов (трёхочковая, трапеция, круги) абсолютные, от L и W зависят
    только периметр, центр и положение второй половины."""
    cy = W / 2
    lines = [np.array([[0, 0], [L, 0], [L, W], [0, W], [0, 0]], float),  # периметр
             np.array([[L / 2, 0], [L / 2, W]], float),                   # центральная линия
             _arc(L / 2, cy, 1.8, 0, 2 * np.pi)]                          # центральный круг
    bx, r3 = 1.575, 6.75                    # центр кольца от лицевой, радиус трёхочковой
    a = np.arcsin((cy - 0.9) / r3)          # прямые участки трёхочковой: 0.9 м от боковой
    half = [np.array([[0, cy - 2.45], [5.8, cy - 2.45], [5.8, cy + 2.45], [0, cy + 2.45]]),  # трапеция
            _arc(5.8, cy, 1.8, -np.pi / 2, np.pi / 2),                    # штрафной полукруг
            np.vstack([[[0, 0.9]], _arc(bx, cy, r3, -a, a), [[0, W - 0.9]]]),  # трёхочковая
            _arc(bx, cy, 1.25, -np.pi / 2, np.pi / 2)]                    # зона без фола
    for p in half:                          # вторая половина -- зеркально по X
        lines += [p, np.column_stack([L - p[:, 0], p[:, 1]])]
    return lines


def _densify(p, step=0.1):
    """Прямая на корте остаётся прямой на кадре, но точки за горизонтом -> NaN,
    поэтому дробим отрезки, чтобы обрезать их по частям, а не целиком."""
    out = [p[0]]
    for a, b in zip(p[:-1], p[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out += list(a + (b - a) * np.linspace(0, 1, n + 1)[1:, None])
    return np.array(out)


def court_to_minimap(pts, width_m=15.0, px_per_m=40.0, pad_m=1.0):
    """Метры корта (N, 2) -> пиксели схемы draw_court_topdown. Как у вида сверху
    CourtHomography: дальняя боковая вверху, плюс поле pad_m вокруг корта."""
    p = np.atleast_2d(np.asarray(pts, float))
    return np.column_stack([(p[:, 0] + pad_m) * px_per_m, (width_m - p[:, 1] + pad_m) * px_per_m])


def draw_court_topdown(length_m=28.0, width_m=15.0, px_per_m=40.0, pad_m=1.0,
                       floor=(200, 225, 240), line=(40, 40, 40), thickness=2):
    """Схема корта для мини-карты: разметка FIBA на однотонном полу, поле pad_m
    вокруг (там видны игроки за линиями)."""
    w = round((length_m + 2 * pad_m) * px_per_m)
    h = round((width_m + 2 * pad_m) * px_per_m)
    img = np.full((h, w, 3), floor, np.uint8)
    for p in court_lines(length_m, width_m):
        q = court_to_minimap(p, width_m, px_per_m, pad_m)
        cv2.polylines(img, [np.round(q).astype(np.int32)], False, line, thickness, cv2.LINE_AA)
    return img


def draw_markings(ch, frame, color=(255, 0, 255), thickness=4, alpha=0.8):
    """Кадр с подсвеченной разметкой. ch -- CourtHomography, откалиброванная
    под размер именно этого кадра (from_json(..., image_size=(w, h)))."""
    overlay = frame.copy()
    for p in court_lines(ch.length_m, ch.width_m):
        q = ch.court_to_image(_densify(p))
        ok = np.isfinite(q).all(axis=1)
        # За горизонтом точки NaN: рисуем непрерывные видимые куски по отдельности,
        # иначе polylines соединит их отрезком через невидимую часть.
        for run in np.split(q, np.flatnonzero(np.diff(ok)) + 1):
            if len(run) > 1 and np.isfinite(run).all():
                cv2.polylines(overlay, [np.round(run).astype(np.int32)], False,
                              color, thickness, cv2.LINE_AA)
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
