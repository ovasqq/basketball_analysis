"""CLI: калибровка -> отчёт, перевод точек, вид сверху, подсветка разметки.

    python main.py                                   # calibration.json из корня проекта
    python main.py --point 655 447 --frame images/source/frame.jpg
    python main.py --frame images/source/frame.jpg --highlight images/output/frame_lightened.jpg
    python main.py другая_калибровка.json ...

Как строить гомографию -- флаг --method:
    manual     точки, протыканные вручную (calibration.json, tools/click_points.py)
    kalicalib  сеть KaliCalib, 91 точка сетки по полу (torch)
    court18    YOLOv8-pose на 18 точек разметки, HoopVision и др. (ultralytics)
    court33    Roboflow basketball-court-detection-2, 33 точки (inference + ROBOFLOW_API_KEY)

    python main.py --method kalicalib --frame images/source/frame.jpg --highlight images/output/auto.jpg

Пути по умолчанию -- в basketball/paths.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from basketball.paths import CALIBRATION, COURT18_WEIGHTS, KALICALIB_WEIGHTS, OUTPUT_DIR
from basketball.topdown.homography import CourtHomography
from basketball.topdown.markings import draw_markings

METHODS = ("manual", "kalicalib", "court18", "court33")
DEFAULT_WEIGHTS = {"kalicalib": KALICALIB_WEIGHTS, "court18": COURT18_WEIGHTS}


def _imwrite(path, img):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise SystemExit(f"не удалось сохранить: {path}")


def _network(method, weights):
    """Сеть выбранного метода. Импорт здесь: зависимости сети (torch, ultralytics,
    inference) нужны только ей."""
    if method == "court33":
        from basketball.court33 import Court33
        return Court33()
    weights = weights or DEFAULT_WEIGHTS[method]
    if not Path(weights).is_file():
        raise SystemExit(f"нет весов {method}: {weights} (как скачать -- в README)")
    if method == "kalicalib":
        from basketball.kalicalib import KaliCalib
        return KaliCalib(weights)
    from basketball.court18 import Court18
    return Court18(weights)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", nargs="?", default=CALIBRATION,
                    help="файл калибровки для --method manual "
                         "(по умолчанию calibration.json в корне проекта)")
    ap.add_argument("--method", choices=METHODS, default="manual",
                    help="как строить гомографию: ручные точки или одна из сетей (см. выше)")
    ap.add_argument("--point", nargs=2, type=float, action="append", metavar=("X", "Y"),
                    help="пиксель на кадре (можно несколько раз)")
    ap.add_argument("--frame", help="кадр для построения вида сверху / подсветки разметки")
    ap.add_argument("--out", default=OUTPUT_DIR / "topdown.png", help="куда сохранить вид сверху")
    ap.add_argument("--highlight", metavar="OUT",
                    help="сохранить кадр с подсвеченной разметкой (нужен --frame)")
    ap.add_argument("--px-per-m", type=float, default=40.0)
    ap.add_argument("--thresh", type=float, default=None,
                    help="порог RANSAC, пиксели (по умолчанию 3 для manual, 35 при ширине 960 "
                         "для kalicalib, 1%% ширины кадра для court18/court33)")
    ap.add_argument("--min-score", type=float, default=None,
                    help="мин. уверенность точки сети (по умолчанию 0 для kalicalib, "
                         "0.5 для court18/court33)")
    ap.add_argument("--weights", default=None,
                    help="веса сети (по умолчанию из basketball/paths.py; court33 качает сам)")
    a = ap.parse_args()
    if a.highlight and not a.frame:
        ap.error("--highlight требует --frame")
    if a.method != "manual" and not a.frame:
        ap.error(f"--method {a.method} требует --frame")

    frame = None
    if a.frame:
        frame = cv2.imread(a.frame)
        if frame is None:
            raise SystemExit(f"не удалось прочитать кадр: {a.frame}")
    if a.method == "manual":
        # Есть кадр -> калибровка масштабируется под его реальное разрешение.
        size = (frame.shape[1], frame.shape[0]) if frame is not None else None
        ch = CourtHomography.from_json(a.json_path, image_size=size,
                                       thresh_px=3.0 if a.thresh is None else a.thresh)
        print(ch.report())
    else:
        try:
            net = _network(a.method, a.weights)
        except ImportError as e:
            raise SystemExit(f"{a.method}: не установлен пакет {e.name} (см. README)")
        except RuntimeError as e:   # court33 без ROBOFLOW_API_KEY
            raise SystemExit(f"{a.method}: {e}")
        kw = {} if a.min_score is None else {"min_score": a.min_score}
        try:
            ch = net.homography(frame, thresh_px=a.thresh, **kw)
        except ValueError as e:     # меньше 4 точек или вырожденная конфигурация
            raise SystemExit(f"{a.method}: гомография не построена: {e}")
        err_px, _ = ch.errors()
        print(f"{a.method}: точек {len(ch.image_pts)}, инлайеров {int(ch.inliers.sum())}, "
              f"медиана ошибки по инлайерам {np.median(err_px[ch.inliers]):.1f} px")
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
