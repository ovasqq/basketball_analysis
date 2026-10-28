"""Игроки на кадре: детекция людей YOLO (COCO, класс person) -> пиксель «низа ног».

Точка ног -- середина нижней стороны бокса: это точка на полу, поэтому её можно
переводить в метры корта через CourtHomography.image_to_court (для головы или
центра бокса гомография неверна, см. README про мяч). Ошибки метода:
- игрок в прыжке: низ бокса над полом, точка уезжает от камеры;
- ноги закрыты другим игроком: бокс обрезан снизу, точка ближе к камере, чем надо;
- широкий шаг: середина между стопами, а не под центром масс (обычно некритично).

Нужен ultralytics и веса детектора (по умолчанию weights/yolo11s.pt, см. README).

Из кода:
    det = PlayerDetector("weights/yolo11s.pt")
    p = det.detect(frame)          # Players: boxes, scores, feet, ids
    p = det.track(frame)           # то же + id трека (ByteTrack), кадры подавать по порядку
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PERSON = 0     # класс person в COCO


@dataclass
class Players:
    boxes: np.ndarray    # (N, 4) x1, y1, x2, y2 в пикселях кадра
    scores: np.ndarray   # (N,)
    feet: np.ndarray     # (N, 2) середина нижней стороны бокса
    ids: np.ndarray      # (N,) id трека, -1 если трекинга нет или трек ещё не подтверждён

    def __len__(self):
        return len(self.boxes)


def feet_from_boxes(boxes):
    """(N, 4) x1, y1, x2, y2 -> (N, 2) середина нижней стороны."""
    b = np.asarray(boxes, float).reshape(-1, 4)
    return np.column_stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]])


def foot_flags(boxes, image_h, court_m, length_m=28.0, width_m=15.0, edge_px=3, margin_m=2.0):
    """Почему точке ног нельзя верить -- список меток на игрока (пустой: всё в порядке).
    cut       -- бокс касается нижнего края кадра: ноги обрезаны, точка выше пола;
    off_court -- точка дальше margin_m от корта или за горизонтом: зритель, судья у
                 стола, ложная детекция."""
    b = np.asarray(boxes, float).reshape(-1, 4)
    c = np.asarray(court_m, float).reshape(-1, 2)
    flags = []
    for (_, _, _, y2), (x, y) in zip(b, c):
        f = []
        if y2 >= image_h - edge_px:
            f.append("cut")
        if not (np.isfinite(x) and np.isfinite(y)
                and -margin_m <= x <= length_m + margin_m and -margin_m <= y <= width_m + margin_m):
            f.append("off_court")
        flags.append(f)
    return flags


class PlayerDetector:
    def __init__(self, weights, device=None, conf=0.3, imgsz=1280):
        # ultralytics нужен только здесь. .pt -- pickle: берите официальные веса ultralytics.
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        if device is None:      # ultralytics сам берёт только cuda/cpu, Apple GPU -- явно
            import torch
            device = ("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
        self.device, self.conf, self.imgsz = device, conf, imgsz

    def _players(self, r):
        b = r.boxes
        if b is None or len(b) == 0:
            z = np.zeros((0, 4))
            return Players(z, np.zeros(0), np.zeros((0, 2)), np.zeros(0, int))
        boxes = b.xyxy.cpu().numpy().astype(np.float64)
        ids = (b.id.cpu().numpy().astype(int) if b.id is not None
               else np.full(len(boxes), -1))
        return Players(boxes, b.conf.cpu().numpy().astype(np.float64),
                       feet_from_boxes(boxes), ids)

    def detect(self, frame_bgr):
        """Люди на одном кадре, без трекинга. imgsz=1280: игроки на дальней стороне
        мелкие, при стандартных 640 их часто не видно."""
        r = self.model.predict(frame_bgr, classes=[PERSON], conf=self.conf, imgsz=self.imgsz,
                               device=self.device, verbose=False)[0]
        return self._players(r)

    def track(self, frame_bgr):
        """То же с id трека (ByteTrack). Кадры видео подавать по порядку: состояние
        трекера живёт внутри модели между вызовами."""
        r = self.model.track(frame_bgr, classes=[PERSON], conf=self.conf, imgsz=self.imgsz,
                             device=self.device, persist=True, tracker="bytetrack.yaml",
                             verbose=False)[0]
        return self._players(r)
