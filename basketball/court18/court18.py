"""Ключевые точки корта моделью YOLOv8-pose на 18 точек: кадр -> пары точек (пиксель, метры).

Модель из туториала «basketball analysis» и проектов на его основе
(github.com/Perceptron04/HoopVision, github.com/HanaFEKI/AI_BasketBall_Analysis_v1,
github.com/j7yn/nba-vision): одна детекция «корт» с 18 ключевыми точками на
линиях разметки (углы, края трапеций, концы средней линии, точки трёхочковой у
лицевых). Обучена на Roboflow-датасете reloc2-den7l (трансляции NBA).

Архитектура -- стандартная ultralytics YOLOv8-pose, поэтому кода сети здесь нет:
нужен пакет ultralytics (AGPL-3.0) и файл весов .pt (см. README).

Из кода:
    det = Court18("weights/court18_yolov8_pose.pt")
    ch = det.homography(frame)        # CourtHomography
"""
from __future__ import annotations

import numpy as np

from basketball.topdown.homography import CourtHomography

# Шаблон точек в порядке выходов модели, метры на корте 28x15. Как в
# TacticalViewConverter.key_points проектов выше: X вдоль длины, y_их -- от верхнего
# края тактической схемы. Цифры их, не FIBA (трапеция 5.18..10 м -- ближе к NBA):
# модель училась именно на них. Направление осей по кадру не важно:
# CourtHomography.from_keypoints приводит его к нашей конвенции.
TEMPLATE_28x15 = np.array([
    (0, 0), (0, 0.91), (0, 5.18), (0, 10), (0, 14.1), (0, 15),      # 0-5   левая лицевая
    (14, 15), (14, 0),                                              # 6-7   средняя линия
    (5.79, 5.18), (5.79, 10),                                       # 8-9   левая штрафная
    (28, 15), (28, 14.1), (28, 10), (28, 5.18), (28, 0.91), (28, 0),  # 10-15 правая лицевая
    (22.21, 5.18), (22.21, 10),                                     # 16-17 правая штрафная
], float)


def template_court_m(length_m=28.0, width_m=15.0):
    """Шаблон в наших метрах: Y = W - y_их (верх схемы -- дальняя боковая),
    пропорционально растянутый под размер корта."""
    t = TEMPLATE_28x15 * [length_m / 28.0, width_m / 15.0]
    return np.column_stack([t[:, 0], width_m - t[:, 1]])


class Court18:
    def __init__(self, weights, device=None):
        # ultralytics нужен только здесь. Внимание: .pt -- это pickle, загрузка
        # исполняет код из файла. Берите веса только из доверенного источника.
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.device = device
        kpt = (getattr(self.model.model, "yaml", None) or {}).get("kpt_shape")
        if self.model.task != "pose" or not kpt or kpt[0] != len(TEMPLATE_28x15):
            raise ValueError(f"{weights}: ожидается pose-модель на {len(TEMPLATE_28x15)} точек, "
                             f"а это task={self.model.task}, kpt_shape={kpt}")

    def keypoints(self, frame_bgr, length_m=28.0, width_m=15.0, box_conf=0.25):
        """-> (image_px (N, 2), court_m (N, 2), scores (N,)) для найденных точек
        самой уверенной детекции корта. Ненайденные точки ultralytics отдаёт как ровно (0, 0)."""
        r = self.model.predict(frame_bgr, conf=box_conf, device=self.device, verbose=False)[0]
        empty = np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)
        if r.keypoints is None or len(r.keypoints.xy) == 0:
            return empty
        best = int(r.boxes.conf.argmax()) if r.boxes is not None and len(r.boxes) else 0
        xy = r.keypoints.xy[best].cpu().numpy().astype(np.float64)
        conf = (r.keypoints.conf[best].cpu().numpy() if r.keypoints.conf is not None
                else np.ones(len(xy)))
        if len(xy) != len(TEMPLATE_28x15):
            raise ValueError(f"модель отдаёт {len(xy)} точек, а шаблон на {len(TEMPLATE_28x15)}: "
                             "это не 18-точечная модель корта")
        found = (xy != 0).any(axis=1)
        grid = template_court_m(length_m, width_m)
        return xy[found], grid[found], conf[found].astype(np.float64)

    def homography(self, frame_bgr, min_score=0.5, length_m=28.0, width_m=15.0, thresh_px=None):
        """CourtHomography по точкам с уверенностью >= min_score. Порог RANSAC по
        умолчанию -- 1 % ширины кадра (19 px при 1920)."""
        img, crt, scores = self.keypoints(frame_bgr, length_m, width_m)
        keep = scores >= min_score
        if thresh_px is None:
            thresh_px = 0.01 * frame_bgr.shape[1]
        names = [f"kp{i}" for i in range(int(keep.sum()))]
        return CourtHomography.from_keypoints(img[keep], crt[keep], length_m, width_m,
                                              thresh_px=thresh_px, names=names)
