"""Проверки сетей ключевых точек без весов: шаблоны, разбор выхода, приведение осей.
Запуск из корня проекта: python -m tests.test_court_nets"""
from __future__ import annotations

from types import SimpleNamespace as NS

import numpy as np

from basketball.court18 import Court18, template_court_m as t18
from basketball.court33 import Court33, template_court_m as t33
from basketball.court33.court33 import AIR_POINTS
from basketball.paths import CALIBRATION
from basketball.topdown.homography import CourtHomography

FRAME = np.zeros((1280, 1920, 3), np.uint8)   # размер, под который сделан calibration.json
TRUE = CourtHomography.from_json(CALIBRATION)  # «настоящая» камера для синтетики


def test_templates_inside_court():
    for t, n in ((t18(), 18), (t33(), 33)):
        assert t.shape == (n, 2)
        assert (t >= 0).all() and (t[:, 0] <= 28).all() and (t[:, 1] <= 15).all()
    # углы корта присутствуют в обоих шаблонах
    for t in (t18(), t33()):
        for corner in ([0, 0], [28, 0], [0, 15], [28, 15]):
            assert np.isclose(t, corner).all(axis=1).any()


def test_from_keypoints_fixes_mirrored_template():
    crt = t33()[[0, 5, 27, 32, 9, 11, 21, 23, 15, 17]]
    img = TRUE.court_to_image(crt)
    pts = np.array([[14, 7.5], [3, 2], [25, 12]])
    for mirrored in (crt, [28, 0] + crt * [-1, 1], [0, 15] + crt * [1, -1], [28, 15] - crt):
        ch = CourtHomography.from_keypoints(img, mirrored)
        assert np.allclose(ch.court_to_image(pts), TRUE.court_to_image(pts), atol=1e-3)


class _T(np.ndarray):
    """Минимальная замена torch.Tensor: .cpu().numpy()."""
    def __new__(cls, a):
        return np.asarray(a, float).view(cls)

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)


class _Boxes:
    def __init__(self, conf):
        self.conf = _T(conf)

    def __len__(self):
        return len(self.conf)


def _court18(xy, conf):
    """Court18 с подменённой моделью ultralytics: две детекции, нужная -- вторая."""
    res = NS(keypoints=NS(xy=[_T(xy * 0 + 5), _T(xy)], conf=[_T(conf), _T(conf)]),
             boxes=_Boxes([0.3, 0.9]))
    det = Court18.__new__(Court18)
    det.model, det.device = NS(predict=lambda *a, **k: [res]), None
    return det


def test_court18_parses_ultralytics_output():
    grid = t18()
    xy = TRUE.court_to_image(grid)
    conf = np.full(18, 0.9)
    xy[3] = 0                                   # ненайденная точка -> (0, 0)
    conf[7] = 0.1                               # неуверенная точка
    img, crt, _ = _court18(xy, conf).keypoints(FRAME)
    assert len(img) == 17 and np.allclose(crt, np.delete(grid, 3, axis=0))
    ch = _court18(xy, conf).homography(FRAME)
    assert len(ch.image_pts) == 16
    assert np.allclose(ch.court_to_image([14, 7.5]), TRUE.court_to_image([14, 7.5]), atol=1e-3)


def test_court18_rejects_wrong_model():
    import importlib.util
    if importlib.util.find_spec("ultralytics") is None:
        print("    пропущен: нет ultralytics")
        return
    from unittest import mock

    for task, kpt in (("detect", None), ("pose", [17, 3])):
        fake = NS(task=task, model=NS(yaml={"kpt_shape": kpt}))
        with mock.patch("ultralytics.YOLO", return_value=fake):
            try:
                Court18("x.pt")
            except ValueError:
                continue
        raise AssertionError(f"не отвергнута модель task={task}, kpt_shape={kpt}")


def test_court33_parses_inference_output():
    grid = t33()
    xy = TRUE.court_to_image(grid)
    kps = [NS(x=x, y=y, confidence=0.9, class_id=i) for i, (x, y) in enumerate(xy)
           if i not in (12, 13)]                # часть точек сеть не нашла
    weak = NS(keypoints=kps[:4], confidence=0.2)
    strong = NS(keypoints=kps, confidence=0.95)
    det = Court33.__new__(Court33)
    det.model = NS(infer=lambda *a, **k: [NS(predictions=[weak, strong])])
    img, crt, _ = det.keypoints(FRAME)
    assert len(img) == 33 - 2 - len(AIR_POINTS)
    ch = det.homography(FRAME)
    assert np.allclose(ch.court_to_image([14, 7.5]), TRUE.court_to_image([14, 7.5]), atol=1e-3)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
