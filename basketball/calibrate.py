"""Калибровка корта выбранным способом: кадр -> CourtHomography.

Способы (--method в main.py и video.py):
    manual     точки, протыканные вручную (calibration.json, tools/click_points.py)
    kalicalib  сеть KaliCalib, 91 точка сетки по полу (torch)
    court18    YOLOv8-pose на 18 точек разметки (ultralytics)
    court33    Roboflow basketball-court-detection-2, 33 точки (inference + ROBOFLOW_API_KEY)

Из кода:
    ch, info = court_homography("kalicalib", frame)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from basketball.paths import CALIBRATION, COURT18_WEIGHTS, KALICALIB_WEIGHTS
from basketball.topdown.homography import CourtHomography

METHODS = ("manual", "kalicalib", "court18", "court33")
DEFAULT_WEIGHTS = {"kalicalib": KALICALIB_WEIGHTS, "court18": COURT18_WEIGHTS}


class CalibrationError(Exception):
    """Калибровку не получить: нет весов или пакета, мало точек и т.п. Текст -- для человека."""


def network(method, weights=None):
    """Сеть выбранного метода. Импорт здесь: зависимости сети (torch, ultralytics,
    inference) нужны только ей."""
    try:
        if method == "court33":
            from basketball.court33 import Court33
            return Court33()
        weights = weights or DEFAULT_WEIGHTS[method]
        if not Path(weights).is_file():
            raise CalibrationError(f"нет весов {method}: {weights} (как скачать -- в README)")
        if method == "kalicalib":
            from basketball.kalicalib import KaliCalib
            return KaliCalib(weights)
        from basketball.court18 import Court18
        return Court18(weights)
    except ImportError as e:
        raise CalibrationError(f"{method}: не установлен пакет {e.name} (см. README)") from e
    except RuntimeError as e:   # court33 без ROBOFLOW_API_KEY
        raise CalibrationError(f"{method}: {e}") from e


def court_homography(method, frame=None, json_path=CALIBRATION, thresh_px=None,
                     min_score=None, weights=None):
    """-> (CourtHomography, отчёт строкой). Для manual кадр нужен только чтобы
    подогнать калибровку под его разрешение; сетям кадр обязателен."""
    if method not in METHODS:
        raise CalibrationError(f"неизвестный способ калибровки: {method}")
    if method == "manual":
        size = (frame.shape[1], frame.shape[0]) if frame is not None else None
        ch = CourtHomography.from_json(json_path, image_size=size,
                                       thresh_px=3.0 if thresh_px is None else thresh_px)
        return ch, ch.report()
    if frame is None:
        raise CalibrationError(f"{method}: нужен кадр")
    net = network(method, weights)
    kw = {} if min_score is None else {"min_score": min_score}
    try:
        ch = net.homography(frame, thresh_px=thresh_px, **kw)
    except ValueError as e:     # меньше 4 точек или вырожденная конфигурация
        raise CalibrationError(f"{method}: гомография не построена: {e}") from e
    err_px, _ = ch.errors()
    return ch, (f"{method}: точек {len(ch.image_pts)}, инлайеров {int(ch.inliers.sum())}, "
                f"медиана ошибки по инлайерам {np.median(err_px[ch.inliers]):.1f} px")
