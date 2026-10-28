"""Чтение видео по кадрам: номер кадра, время в секундах, кадр BGR.

Время берётся из таймстемпа контейнера (CAP_PROP_POS_MSEC), а не frame / fps:
в записях бывают пропущенные кадры и переменный fps.

Из кода:
    for idx, t, frame in FrameSource("ace_4.mp4", start=30000, count=900, stride=2):
        ...
"""
from __future__ import annotations

from pathlib import Path

import cv2


class FrameSource:
    def __init__(self, path, start=0, count=None, stride=1):
        if stride < 1:
            raise ValueError(f"stride должен быть >= 1, получено {stride}")
        self.path, self.start, self.count, self.stride = str(path), int(start), count, int(stride)
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise FileNotFoundError(f"не удалось открыть видео: {path}")
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    @property
    def name(self):
        return Path(self.path).stem

    def __len__(self):
        """Сколько кадров отдаст итерация (оценка по CAP_PROP_FRAME_COUNT)."""
        n = max(0, self.total - self.start) if self.total > 0 else 0
        if self.count is not None:
            n = min(n, self.count) if n else self.count
        return -(-n // self.stride)

    def __iter__(self):
        cap = cv2.VideoCapture(self.path)
        try:
            if self.start:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.start)
            idx, end = self.start, (self.start + self.count if self.count is not None else None)
            while end is None or idx < end:
                if (idx - self.start) % self.stride:
                    if not cap.grab():      # пропускаемый кадр: без декодирования в BGR
                        return
                else:
                    ok, frame = cap.read()
                    if not ok:
                        return
                    t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    yield idx, t, frame
                idx += 1
        finally:
            cap.release()
