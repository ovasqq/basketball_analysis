"""Быстрые проверки калибровки. Запуск из корня проекта: python -m tests.test_homography"""
from __future__ import annotations

import warnings

import numpy as np

from basketball.paths import CALIBRATION as JSON
from basketball.topdown.homography import CourtHomography


def test_same_aspect_no_warning_and_scales():
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # любое предупреждение -> падение теста
        full = CourtHomography.from_json(JSON)                      # 1920x1280 из JSON
        half = CourtHomography.from_json(JSON, image_size=(960, 640))
    # Та же точка корта на кадре вдвое меньше -> ровно вдвое меньшие пиксели.
    assert np.allclose(half.court_to_image([14, 7.5]), full.court_to_image([14, 7.5]) / 2)


def test_other_aspect_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        CourtHomography.from_json(JSON, image_size=(1920, 1080))
    assert any("соотношение сторон" in str(x.message) for x in w)


def test_round_trip():
    ch = CourtHomography.from_json(JSON)
    pts = np.array([[0, 0], [28, 15], [5.8, 7.5], [14, 3]], float)
    assert np.allclose(ch.image_to_court(ch.court_to_image(pts)), pts, atol=1e-6)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
