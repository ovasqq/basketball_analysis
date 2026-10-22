"""Ключевые точки корта моделью Roboflow basketball-court-detection-2 (33 точки).

Её используют roboflow/sports (ветка feat/basketball),
github.com/cs25m116/Basketball-Player-Detection-Tracking-Identification и
github.com/elijahzhao24/nba_3d_scene_reconstruction. Точки -- вершины разметки
из sports/basketball/config.py (CourtConfiguration._raw_vertices_centimeters):
углы, края трапеций и трёхочковой, отметки линии вбрасывания, середины линий.

Веса не публикуются файлом: пакет inference (Roboflow) скачивает их по model_id
с API-ключом (ROBOFLOW_API_KEY) и дальше считает локально. Обучена на трансляциях NBA.

Из кода:
    det = Court33()                   # ключ из ROBOFLOW_API_KEY
    ch = det.homography(frame)        # CourtHomography
"""
from __future__ import annotations

import os

import numpy as np

from basketball.topdown.homography import CourtHomography

MODEL_ID = "basketball-court-detection-2/14"

# Размеры FIBA из PRESETS_CENTIMETERS[League.FIBA] в roboflow/sports, метры.
# Модель училась на площадках NBA, но точки -- это одни и те же элементы разметки,
# поэтому на корте FIBA берём их положение по FIBA.
FIBA = dict(three_r=6.75, three_straight=3.30, side_to_three=0.90, paint_w=4.90,
            paint_l=5.80, rim=1.575, throw_line=8.30)

# Вершины 6 и 26 -- точки «под кольцом»: на полу их не видно, а сеть, скорее всего,
# ставит их на само кольцо (в воздухе). Для гомографии пола не годятся.
AIR_POINTS = (6, 26)


def template_court_m(length_m=28.0, width_m=15.0, f=FIBA):
    """33 вершины в порядке выходов модели (class_id), метры. Оси как в sports:
    X вдоль длины, Y от боковой с вершиной 0. Какая это боковая на кадре,
    определяет CourtHomography.from_keypoints."""
    L, W = length_m, width_m
    p0 = (W - f["paint_w"]) / 2
    p1, mid = p0 + f["paint_w"], W / 2
    s3, st = f["side_to_three"], f["three_straight"]
    arc = f["rim"] + f["three_r"]
    tl = f["throw_line"]
    return np.array([
        (0, 0), (0, s3), (0, p0), (0, p1), (0, W - s3), (0, W),      # 0-5   левая лицевая
        (f["rim"], mid),                                             # 6     под левым кольцом
        (st, s3), (st, W - s3),                                      # 7-8   концы прямых трёхочковой
        (f["paint_l"], p0), (f["paint_l"], mid), (f["paint_l"], p1),  # 9-11  штрафная
        (tl, 0), (arc, mid), (tl, W),                                # 12-14
        (L / 2, 0), (L / 2, mid), (L / 2, W),                        # 15-17 средняя линия
        (L - tl, 0), (L - arc, mid), (L - tl, W),                    # 18-20
        (L - f["paint_l"], p0), (L - f["paint_l"], mid), (L - f["paint_l"], p1),  # 21-23
        (L - st, s3), (L - st, W - s3),                              # 24-25
        (L - f["rim"], mid),                                         # 26    под правым кольцом
        (L, 0), (L, s3), (L, p0), (L, p1), (L, W - s3), (L, W),      # 27-32 правая лицевая
    ], float)


class Court33:
    def __init__(self, model_id=MODEL_ID, api_key=None):
        # inference нужен только здесь; ключ -- только чтобы скачать веса.
        from inference import get_model

        api_key = api_key or os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            raise RuntimeError("нужен ключ Roboflow: переменная окружения ROBOFLOW_API_KEY")
        self.model = get_model(model_id=model_id, api_key=api_key)

    def keypoints(self, frame_bgr, length_m=28.0, width_m=15.0, conf=0.3):
        """-> (image_px (N, 2), court_m (N, 2), scores (N,)) самой уверенной детекции
        корта, без точек «под кольцом»."""
        preds = self.model.infer(frame_bgr, confidence=conf)[0].predictions
        if not preds:
            return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)
        best = max(preds, key=lambda p: p.confidence)
        grid = template_court_m(length_m, width_m)
        kps = [k for k in best.keypoints if k.class_id not in AIR_POINTS]
        img = np.array([[k.x, k.y] for k in kps], float).reshape(-1, 2)
        crt = grid[[k.class_id for k in kps]].reshape(-1, 2)
        return img, crt, np.array([k.confidence for k in kps], float)

    def homography(self, frame_bgr, min_score=0.5, length_m=28.0, width_m=15.0, thresh_px=None):
        """CourtHomography по точкам с уверенностью >= min_score (0.5, как
        KEYPOINT_DETECTION_MODEL_ANCHOR_CONFIDENCE у Roboflow). Порог RANSAC по
        умолчанию -- 1 % ширины кадра."""
        img, crt, scores = self.keypoints(frame_bgr, length_m, width_m)
        keep = scores >= min_score
        if thresh_px is None:
            thresh_px = 0.01 * frame_bgr.shape[1]
        names = [f"kp{i}" for i in range(int(keep.sum()))]
        return CourtHomography.from_keypoints(img[keep], crt[keep], length_m, width_m,
                                              thresh_px=thresh_px, names=names)
