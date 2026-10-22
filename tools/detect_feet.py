"""Пиксели «низа ног» игроков на изображениях. Одна строка JSON на кадр (JSON Lines).

    python -m tools.detect_feet images/source/frame.jpg
    python -m tools.detect_feet кадры/*.jpg --calibration calibration.json --draw images/output/feet

С --calibration к каждой точке добавляются метры корта (court_m; null, если точка за
горизонтом). С --draw рядом сохраняются кадры с боксами и точками ног.
Запуск из корня проекта.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from basketball.paths import PLAYERS_WEIGHTS
from basketball.players import PlayerDetector
from basketball.topdown.homography import CourtHomography


def draw_players(frame, players, color=(0, 255, 255)):
    out = frame.copy()
    for (x1, y1, x2, y2), (fx, fy), i in zip(players.boxes, players.feet, players.ids):
        cv2.rectangle(out, (round(x1), round(y1)), (round(x2), round(y2)), color, 2)
        cv2.circle(out, (round(fx), round(fy)), 6, (0, 0, 255), -1)
        if i >= 0:
            cv2.putText(out, str(i), (round(x1), round(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--weights", default=PLAYERS_WEIGHTS, help="веса детектора людей")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--calibration", help="calibration.json: добавить метры корта")
    ap.add_argument("--draw", metavar="DIR", help="сохранить кадры с боксами и точками ног")
    a = ap.parse_args()
    if not Path(a.weights).is_file():
        raise SystemExit(f"нет весов детектора: {a.weights} (как скачать -- в README)")

    det = PlayerDetector(a.weights, conf=a.conf)
    for path in a.images:
        frame = cv2.imread(path)
        if frame is None:
            raise SystemExit(f"не удалось прочитать кадр: {path}")
        h, w = frame.shape[:2]
        p = det.detect(frame)
        court = None
        if a.calibration:
            court = CourtHomography.from_json(a.calibration, image_size=(w, h)).image_to_court(p.feet)
        rows = []
        for k in range(len(p)):
            row = {"bbox": np.round(p.boxes[k], 1).tolist(), "score": round(float(p.scores[k]), 3),
                   "foot_px": np.round(p.feet[k], 1).tolist()}
            if court is not None:
                row["court_m"] = (np.round(court[k], 2).tolist()
                                  if np.isfinite(court[k]).all() else None)
            rows.append(row)
        print(json.dumps({"image": path, "image_w": w, "image_h": h, "players": rows},
                         ensure_ascii=False))
        if a.draw:
            out = Path(a.draw) / Path(path).name
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), draw_players(frame, p))


if __name__ == "__main__":
    main()
