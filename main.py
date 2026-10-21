"""CLI: калибровка -> отчёт, перевод точек, вид сверху, подсветка разметки.

    python main.py                                   # calibration.json из корня проекта
    python main.py --point 655 447 --frame images/source/frame.jpg
    python main.py --frame images/source/frame.jpg --highlight images/output/frame_lightened.jpg
    python main.py другая_калибровка.json ...
    python main.py --auto --frame images/source/frame.jpg --highlight images/output/auto.jpg   # KaliCalib

Пути по умолчанию -- в basketball/paths.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from basketball.homography import CourtHomography
from basketball.markings import draw_markings
from basketball.paths import CALIBRATION, KALICALIB_WEIGHTS, OUTPUT_DIR


def _imwrite(path, img):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise SystemExit(f"не удалось сохранить: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", nargs="?", default=CALIBRATION,
                    help="файл калибровки (по умолчанию calibration.json в корне проекта)")
    ap.add_argument("--point", nargs=2, type=float, action="append", metavar=("X", "Y"),
                    help="пиксель на кадре (можно несколько раз)")
    ap.add_argument("--frame", help="кадр для построения вида сверху / подсветки разметки")
    ap.add_argument("--out", default=OUTPUT_DIR / "topdown.png", help="куда сохранить вид сверху")
    ap.add_argument("--highlight", metavar="OUT",
                    help="сохранить кадр с подсвеченной разметкой (нужен --frame)")
    ap.add_argument("--px-per-m", type=float, default=40.0)
    ap.add_argument("--thresh", type=float, default=None,
                    help="порог RANSAC, пиксели (по умолчанию 3 для ручных точек, "
                         "35 при ширине 960 для --auto)")
    ap.add_argument("--auto", action="store_true",
                    help="калибровка нейросетью KaliCalib по --frame вместо json_path (нужен torch)")
    ap.add_argument("--weights", default=KALICALIB_WEIGHTS, help="веса KaliCalib для --auto")
    a = ap.parse_args()
    if a.highlight and not a.frame:
        ap.error("--highlight требует --frame")
    if a.auto and not a.frame:
        ap.error("--auto требует --frame")

    frame = None
    if a.frame:
        frame = cv2.imread(a.frame)
        if frame is None:
            raise SystemExit(f"не удалось прочитать кадр: {a.frame}")
    if a.auto:
        # Импорт здесь: без --auto torch не нужен.
        from basketball.kalicalib import KaliCalib
        if not Path(a.weights).is_file():
            raise SystemExit(f"нет весов KaliCalib: {a.weights} (как скачать -- в README)")
        ch = KaliCalib(a.weights).homography(frame, thresh_px=a.thresh)
        err_px, _ = ch.errors()
        print(f"KaliCalib: точек {len(ch.image_pts)}, инлайеров {int(ch.inliers.sum())}, "
              f"медиана ошибки по инлайерам {np.median(err_px[ch.inliers]):.1f} px")
    else:
        # Есть кадр -> калибровка масштабируется под его реальное разрешение.
        size = (frame.shape[1], frame.shape[0]) if frame is not None else None
        ch = CourtHomography.from_json(a.json_path, image_size=size,
                                       thresh_px=3.0 if a.thresh is None else a.thresh)
        print(ch.report())
    tops = []
    for x, y in a.point or []:
        c = ch.image_to_court([x, y])
        t = ch.image_to_topdown([x, y], a.px_per_m)
        tops.append(t)
        print(f"пиксель ({x:.1f}, {y:.1f}) -> корт ({c[0]:.2f}, {c[1]:.2f}) м "
              f"-> вид сверху ({t[0]:.0f}, {t[1]:.0f}) px")
    if frame is not None:
        topdown = ch.warp_topdown(frame, a.px_per_m)
        for t in tops:
            if np.isfinite(t).all():
                cv2.circle(topdown, (round(t[0]), round(t[1])), 5, (0, 0, 255), -1)
        _imwrite(a.out, topdown)
        print(f"вид сверху сохранён: {a.out}")
    if a.highlight:
        _imwrite(a.highlight, draw_markings(ch, frame))
        print(f"подсветка разметки сохранена: {a.highlight}")


if __name__ == "__main__":
    main()
