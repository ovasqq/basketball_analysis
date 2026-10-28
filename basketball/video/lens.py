"""Выпрямление дисторсии объектива. Гомография верна только для кадра без дисторсии:
у широкоугольных камер (как в SportCenter, k1 = -0.35) прямые линии корта на сыром
кадре изогнуты. Поэтому кадр сначала выпрямляется, и всё дальнейшее -- калибровка
корта, детекция, точки ног -- работает на выпрямленном кадре.

Карты пересчёта строятся один раз, на кадр остаётся быстрый cv2.remap.
Матрица камеры после выпрямления та же K (как cv2.undistort): углы кадра с
чёрными полями обрезаются, размер кадра не меняется.

Из кода:
    lens = Lens.from_json("data/sportcenter/calibration.json", "ace_4")
    frame = lens.undistort(raw)
"""
from __future__ import annotations

import json

import cv2
import numpy as np


class Lens:
    def __init__(self, K, dist, image_size):
        self.K = np.asarray(K, np.float64).reshape(3, 3)
        self.dist = np.asarray(dist, np.float64).ravel()
        self.image_size = (int(image_size[0]), int(image_size[1]))   # (w, h)
        self._maps = cv2.initUndistortRectifyMap(self.K, self.dist, None, self.K,
                                                 self.image_size, cv2.CV_16SC2)

    @classmethod
    def from_json(cls, path, camera=None, image_size=None):
        """JSON с полями K и dist -- на верхнем уровне или под ключом камеры
        (формат calibration.json SportCenter: {"ace_0": {"K", "dist", "R", "t"}, ...}).
        Размер кадра в этом формате не хранится -- передайте image_size=(w, h)."""
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if camera is not None:
            if camera not in d:
                raise KeyError(f"{path}: нет камеры {camera!r}, есть: {', '.join(d)}")
            d = d[camera]
        size = image_size or d.get("image_size")
        if size is None:
            raise ValueError(f"{path}: размер кадра не указан, передайте image_size=(w, h)")
        return cls(d["K"], d["dist"], size)

    def undistort(self, frame):
        h, w = frame.shape[:2]
        if (w, h) != self.image_size:
            raise ValueError(f"кадр {w}x{h}, а калибровка объектива для "
                             f"{self.image_size[0]}x{self.image_size[1]}")
        return cv2.remap(frame, *self._maps, cv2.INTER_LINEAR)

    def undistort_points(self, pts):
        """Пиксели сырого кадра -> пиксели выпрямленного (для точек, найденных
        на сыром кадре)."""
        p = np.asarray(pts, np.float64).reshape(-1, 1, 2)
        return cv2.undistortPoints(p, self.K, self.dist, P=self.K).reshape(-1, 2)
