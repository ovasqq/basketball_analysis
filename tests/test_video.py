"""Проверки обработки видео без весов и датасета. Запуск из корня проекта:
python -m tests.test_video"""
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

from basketball.paths import CALIBRATION
from basketball.players import foot_flags
from basketball.topdown import CourtHomography, court_to_minimap, draw_court_topdown
from basketball.video import FrameSource, Lens, Renderer


def _write_video(path, n=10, size=(64, 48)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, size)
    for i in range(n):
        w.write(np.full((size[1], size[0], 3), i * 20, np.uint8))   # номер кадра -- в яркости
    w.release()


def test_frame_source_start_count_stride():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "v.mp4"
        _write_video(p)
        src = FrameSource(p, start=2, count=6, stride=2)
        got = list(src)
        assert [i for i, _, _ in got] == [2, 4, 6] and len(src) == 3
        assert [round(f.mean() / 20) for _, _, f in got] == [2, 4, 6]   # кадр тот самый
        ts = [t for _, t, _ in got]
        assert np.allclose(np.diff(ts), 0.2, atol=0.02)                 # 10 fps, через кадр
        assert len(list(FrameSource(p, start=8))) == 2                  # до конца видео


def test_lens_straightens_distorted_points():
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    lens = Lens(K, [-0.3, 0.1, 0, 0, 0], (640, 480))
    # Точки прямой y = 400 на идеальном кадре искажаем и выпрямляем обратно.
    ideal = np.column_stack([np.linspace(50, 590, 7), np.full(7, 400.0)])
    n = (ideal - [320, 240]) / 500
    r2 = (n ** 2).sum(axis=1, keepdims=True)
    distorted = n * (1 - 0.3 * r2 + 0.1 * r2 ** 2) * 500 + [320, 240]
    assert np.allclose(lens.undistort_points(distorted), ideal, atol=0.5)
    frame = np.zeros((480, 640, 3), np.uint8)
    assert lens.undistort(frame).shape == frame.shape


def test_foot_flags():
    boxes = [[0, 0, 10, 50], [0, 0, 10, 99], [0, 0, 10, 50], [0, 0, 10, 50]]
    court = [[5, 5], [5, 5], [40, 5], [np.nan, np.nan]]
    assert foot_flags(boxes, 100, court) == [[], ["cut"], ["off_court"], ["off_court"]]


def test_minimap_matches_topdown_convention():
    img = draw_court_topdown(28, 15, px_per_m=10, pad_m=1)
    assert img.shape[:2] == (170, 300)
    # угол (0, 0) -- ближний левый -> внизу слева; дальняя боковая вверху
    assert np.allclose(court_to_minimap([[0, 0], [28, 15]], 15, 10, 1), [[10, 160], [290, 10]])


def test_renderer_draws_frame_and_minimap():
    ch = CourtHomography.from_json(CALIBRATION, image_size=(480, 320))
    frame = np.zeros((320, 480, 3), np.uint8)
    foot = ch.court_to_image([14, 7.5])
    rec = {"frame": 0, "t": 0.0, "players": [
        {"id": 3, "bbox": [foot[0] - 5, foot[1] - 30, foot[0] + 5, foot[1]], "foot_px": foot.tolist(),
         "court_m": [14, 7.5], "flags": []},
        {"id": 4, "bbox": [0, 280, 10, 320], "foot_px": [5, 320], "court_m": None, "flags": ["cut"]}]}
    r = Renderer(ch, px_per_m=10)
    out = r.draw(frame, rec)
    assert out.shape[0] == 320 and out.shape[1] > 480
    out = r.draw(frame, {**rec, "frame": 1})
    assert len(r.trails[3]) == 2 and 4 not in r.trails      # флагнутый в хвосты не идёт


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
