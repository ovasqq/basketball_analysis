"""Обработка видео с неподвижной камеры: игроки на схеме корта.

Видео -> выпрямление объектива -> калибровка корта (один раз) -> трекинг игроков ->
точка ног -> метры корта. На выходе <видео>.jsonl (строка на каждый кадр) и
<видео>.meta.json (параметры прогона, гомография). Видео «кадр + мини-карта» по ним
строит отдельно render_video.py.

    python video.py -c configs/video_ace_4.json                 # параметры из конфига
    python video.py -c configs/video_ace_4.json --start 45000   # флаги переопределяют конфиг

    # то же без конфига: SportCenter, камера ace_4, 30 секунд с 30000-го кадра, KaliCalib
    python video.py data/sportcenter/multicam/ace_4.mp4 \\
        --lens data/sportcenter/calibration.json:ace_4 --method kalicalib \\
        --start 30000 --frames 900

    # Выпрямленный кадр для ручной калибровки, потом --method manual --calibration ...
    python video.py data/sportcenter/multicam/ace_4.mp4 \\
        --lens data/sportcenter/calibration.json:ace_4 --start 30000 --dump-frame images/source/ace_4.jpg

Запуск из корня проекта.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from basketball.calibrate import METHODS, CalibrationError, court_homography
from basketball.config import parse_args
from basketball.paths import CALIBRATION, OUTPUT_DIR, PLAYERS_WEIGHTS
from basketball.players import PlayerDetector, foot_flags
from basketball.video import FrameSource, Lens


def _lens(spec, source):
    """'calibration.json:ace_4' или 'lens.json' -> Lens под размер кадров видео."""
    if not spec:
        return None
    path, _, camera = spec.partition(":")
    return Lens.from_json(path, camera or None, image_size=(source.width, source.height))


def _first_frame(source, lens):
    for _, _, frame in FrameSource(source.path, start=source.start, count=1):
        return lens.undistort(frame) if lens else frame
    raise SystemExit(f"в видео нет кадра {source.start}")


def _record(idx, t, frame, players, ch):
    h, w = frame.shape[:2]
    court = ch.image_to_court(players.feet) if len(players) else np.zeros((0, 2))
    flags = foot_flags(players.boxes, h, court, ch.length_m, ch.width_m)
    rows = []
    for k in range(len(players)):
        rows.append({"id": int(players.ids[k]),
                     "bbox": np.round(players.boxes[k], 1).tolist(),
                     "score": round(float(players.scores[k]), 3),
                     "foot_px": np.round(players.feet[k], 1).tolist(),
                     "court_m": (np.round(court[k], 3).tolist()
                                 if np.isfinite(court[k]).all() else None),
                     "flags": flags[k]})
    return {"frame": idx, "t": round(t, 4), "image_w": w, "image_h": h, "players": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?")
    ap.add_argument("--lens", metavar="JSON[:CAMERA]",
                    help="калибровка объектива (K, dist) для выпрямления кадров; "
                         "без неё кадры не выпрямляются")
    ap.add_argument("--method", choices=METHODS, default="manual",
                    help="калибровка корта: ручные точки или сеть (по первому кадру)")
    ap.add_argument("--calibration", default=CALIBRATION,
                    help="calibration.json для --method manual (по выпрямленному кадру)")
    ap.add_argument("--start", type=int, default=0, help="первый кадр")
    ap.add_argument("--frames", type=int, default=None, help="сколько кадров обработать")
    ap.add_argument("--stride", type=int, default=1, help="брать каждый N-й кадр")
    ap.add_argument("--conf", type=float, default=0.3, help="порог детектора игроков")
    ap.add_argument("--weights", default=PLAYERS_WEIGHTS, help="веса детектора игроков")
    ap.add_argument("--positions", help="куда писать jsonl (по умолчанию images/output/<видео>.jsonl)")
    ap.add_argument("--dump-frame", metavar="JPG",
                    help="только сохранить выпрямленный кадр --start (для ручной калибровки) и выйти")
    a = parse_args(ap, required=("video",),
                   paths=("video", "lens", "calibration", "weights", "positions", "dump_frame"))

    source = FrameSource(a.video, start=a.start, count=a.frames, stride=a.stride)
    lens = _lens(a.lens, source)
    first = _first_frame(source, lens)
    if a.dump_frame:
        Path(a.dump_frame).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(a.dump_frame, first)
        print(f"выпрямленный кадр {a.start} сохранён: {a.dump_frame}")
        return

    try:
        ch, info = court_homography(a.method, first, a.calibration)
    except CalibrationError as e:
        raise SystemExit(str(e))
    print(info)
    if not Path(a.weights).is_file():
        raise SystemExit(f"нет весов детектора игроков: {a.weights} (как скачать -- в README)")
    det = PlayerDetector(a.weights, conf=a.conf)

    positions = Path(a.positions or OUTPUT_DIR / f"{source.name}.jsonl")
    positions.parent.mkdir(parents=True, exist_ok=True)
    meta = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
            "fps": source.fps, "court_m": [ch.length_m, ch.width_m],
            "H_img2court": ch.H_img2court.tolist()}
    positions.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))

    t0, n, k = time.perf_counter(), len(source), -1
    with positions.open("w", encoding="utf-8") as f:
        for k, (idx, t, frame) in enumerate(source):
            if lens:
                frame = lens.undistort(frame)
            rec = _record(idx, t, frame, det.track(frame), ch)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (k + 1) % 50 == 0:
                el = time.perf_counter() - t0
                print(f"  {k + 1}/{n} кадров, {el / (k + 1) * 1000:.0f} мс/кадр")
    el = time.perf_counter() - t0
    print(f"готово: {k + 1} кадров за {el:.0f} с ({el / max(k + 1, 1) * 1000:.0f} мс/кадр)")
    print(f"позиции: {positions}")
    print(f"видео с мини-картой: python render_video.py {positions}")


if __name__ == "__main__":
    main()
