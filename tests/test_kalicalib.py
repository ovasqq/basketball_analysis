"""Проверки KaliCalib. Запуск из корня проекта: python -m tests.test_kalicalib
Сквозной тест пропускается, если нет torch или весов."""
from __future__ import annotations

import importlib.util

import cv2
import numpy as np

from basketball.kalicalib import grid_court_m
from basketball.paths import CALIBRATION, KALICALIB_WEIGHTS, SOURCE_DIR
from basketball.topdown.homography import CourtHomography


def test_grid_matches_kalicalib_template():
    g = grid_court_m()
    assert g.shape == (91, 2)
    # 13 столбцов по длине через 28/12 м, 7 рядов по ширине (viewds.getFieldPoints)
    assert np.allclose(np.unique(g[:, 0]), np.arange(13) * 28 / 12)
    assert np.allclose(np.unique(g[:, 1]), [0, 1.75, 3.80, 6.15, 8.80, 11.75, 15.0])
    assert np.allclose(g[0], [0, 0])            # точка 0 -- ближний левый угол в наших осях


def test_network_close_to_manual_calibration():
    if importlib.util.find_spec("torch") is None or not KALICALIB_WEIGHTS.is_file():
        print("    пропущен: нет torch или весов KaliCalib")
        return
    from basketball.kalicalib import KaliCalib

    frame = cv2.imread(str(SOURCE_DIR / "frame.jpg"))
    auto = KaliCalib(KALICALIB_WEIGHTS).homography(frame)
    manual = CourtHomography.from_json(CALIBRATION, image_size=(frame.shape[1], frame.shape[0]))
    pts = np.array([[14, 7.5], [0, 15], [28, 15], [14, 0]])
    d = np.linalg.norm(auto.court_to_image(pts) - manual.court_to_image(pts), axis=1)
    # На frame.jpg сейчас 12-45 px; порог с запасом -- ловит поломку осей/масштаба (сотни px).
    assert d.max() < 80, d


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
