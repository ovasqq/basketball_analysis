"""Проверки детектора игроков без весов. Запуск из корня проекта: python -m tests.test_players"""
from __future__ import annotations

from types import SimpleNamespace as NS

import numpy as np

from basketball.players import PlayerDetector, feet_from_boxes


def test_feet_is_bottom_center():
    f = feet_from_boxes([[100, 50, 140, 250], [0, 0, 10, 20]])
    assert np.allclose(f, [[120, 250], [5, 20]])
    assert feet_from_boxes(np.zeros((0, 4))).shape == (0, 2)


class _T(np.ndarray):
    """Минимальная замена torch.Tensor: .cpu().numpy()."""
    def __new__(cls, a):
        return np.asarray(a, float).view(cls)

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)


class _Boxes:
    def __init__(self, xyxy, conf, ids):
        self.xyxy, self.conf = _T(xyxy), _T(conf)
        self.id = None if ids is None else _T(ids)

    def __len__(self):
        return len(self.xyxy)


def _detector(boxes):
    det = PlayerDetector.__new__(PlayerDetector)
    det.device, det.conf, det.imgsz = None, 0.3, 1280
    res = NS(boxes=boxes)
    det.model = NS(predict=lambda *a, **k: [res], track=lambda *a, **k: [res])
    return det


def test_detect_and_track_parse_ultralytics_output():
    p = _detector(_Boxes([[10, 20, 30, 120]], [0.9], None)).detect(None)
    assert np.allclose(p.feet, [[20, 120]]) and p.ids.tolist() == [-1]
    p = _detector(_Boxes([[10, 20, 30, 120], [50, 0, 70, 40]], [0.9, 0.5], [3, 7])).track(None)
    assert p.ids.tolist() == [3, 7] and len(p) == 2
    assert len(_detector(_Boxes(np.zeros((0, 4)), [], None)).detect(None)) == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
