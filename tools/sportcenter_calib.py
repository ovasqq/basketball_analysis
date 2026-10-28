"""Калибровка корта для камер датасета SportCenter (EPFL) из их калибровки камер.

В data/sportcenter/calibration.json у каждой камеры K, dist, R, t. Мировая система:
метры, начало -- центр корта (центр круга), пол -- Z = 0, ось Y вдоль корта,
-X -- к дальней (от ace_4) боковой. Гомография пола на выпрямленном кадре:
H = K [r1 r2 t]. Переводим её в наши оси корта и пишем в формате calibration.json
(точки сетки корта, видимые камерой), чтобы дальше работал --method manual.

Размер корта в этом зале нестандартный: 24.5 x 13.1 м. Получен по разметке на кадре
ace_4: штрафные линии на ±6.44 м от центра (+5.8 м по FIBA до лицевой), дальняя
боковая на 6.55 м; трапеция 4.9 м и трёхочковая 6.75 м при этом совпадают с FIBA.

    python -m tools.sportcenter_calib ace_4                 # -> calibrations/sportcenter_ace_4.json
    python -m tools.sportcenter_calib --all
    python -m tools.sportcenter_calib -c configs/sportcenter_calib_all.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from basketball.config import parse_args
from basketball.paths import ROOT

DATASET = ROOT / "data" / "sportcenter"
OUT_DIR = ROOT / "calibrations"
LENGTH_M, WIDTH_M = 24.5, 13.1


def court_to_world(length_m=LENGTH_M, width_m=WIDTH_M):
    """3x3: наши метры корта (x вдоль, y от камеры ace_4) -> мир SportCenter (Xw, Yw)."""
    return np.array([[0.0, -1.0, width_m / 2], [1.0, 0.0, -length_m / 2], [0.0, 0.0, 1.0]])


def write_calibration(camera, out=None, step_m=1.0):
    cams = json.loads((DATASET / "calibration.json").read_text())
    c = cams[camera]
    cap = cv2.VideoCapture(str(DATASET / "multicam" / f"{camera}.mp4"))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not w:
        raise SystemExit(f"не удалось открыть видео камеры {camera}")
    K, R, t = np.array(c["K"]), np.array(c["R"]), np.array(c["t"]).reshape(3)
    H = K @ np.column_stack([R[:, 0], R[:, 1], t]) @ court_to_world()   # корт -> выпрямленный кадр

    points = []
    for x in np.arange(0, LENGTH_M + 1e-9, step_m):
        for y in np.arange(0, WIDTH_M + 1e-9, step_m):
            p = H @ [x, y, 1.0]
            if p[2] <= 0:
                continue                    # за горизонтом камеры
            u, v = p[:2] / p[2]
            if 0 <= u < w and 0 <= v < h:
                points.append({"name": f"g{x:g}_{y:g}", "image_rel": [round(u / w, 6), round(v / h, 6)],
                               "court_m": [round(float(x), 3), round(float(y), 3)], "role": "fit"})
    if len(points) < 4:
        raise SystemExit(f"{camera}: в кадре меньше 4 точек корта")
    d = {"_comment": f"Камера {camera} датасета SportCenter: точки сетки корта, спроецированные "
                     "калибровкой камер датасета (K, R, t) на ВЫПРЯМЛЕННЫЙ кадр "
                     "(video.py --lens data/sportcenter/calibration.json:" + camera + "). "
                     "Сгенерировано tools/sportcenter_calib.py.",
         "court": {"length_m": LENGTH_M, "width_m": WIDTH_M,
                   "axes": "X вдоль корта, Y поперёк; начало -- угол корта, ближний к камере ace_4 и "
                           "левый на её кадре. Мир SportCenter: Xw = W/2 - y, Yw = x - L/2."},
         "image": {"source": f"data/sportcenter/multicam/{camera}.mp4 (выпрямленный)",
                   "width_px": w, "height_px": h},
         "points": points}
    out = Path(out or OUT_DIR / f"sportcenter_{camera}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d, ensure_ascii=False, indent=1))
    print(f"{camera}: {len(points)} точек -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("camera", nargs="?", help="ace_0 ... ace_7")
    ap.add_argument("--all", action="store_true", help="все камеры")
    a = parse_args(ap)
    if not a.all and not a.camera:
        ap.error("укажите камеру или --all")
    cams = json.loads((DATASET / "calibration.json").read_text())
    for cam in (sorted(cams) if a.all else [a.camera]):
        write_calibration(cam)


if __name__ == "__main__":
    main()
