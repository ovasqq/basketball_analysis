"""Автокалибровка корта нейросетью KaliCalib: кадр -> пары точек (пиксель, метры корта).

Сеть (ResNet18 + декодер) по кадру 960x540 предсказывает 94 тепловые карты в 1/4
разрешения: 91 ключевая точка на сетке 13x7 по полу корта, 2 центра колец
(в воздухе, для гомографии не годятся) и фон. Точки отдаются в CourtHomography --
ровно как ручные клики из calibration.json.

Нужен torch (requirements.txt). Веса -- см. README.

Из кода:
    kc = KaliCalib("weights/kalicalib_challenge.pth")
    ch = kc.homography(frame)          # CourtHomography, как from_json
"""
from __future__ import annotations

import cv2
import numpy as np

from basketball.topdown.homography import CourtHomography

NET_W, NET_H = 960, 540        # вход сети
STRIDE = 4                     # выход сети в 1/4 разрешения входа
N_GRID = 91                    # точки сетки на полу; каналы 91, 92 -- кольца, 93 -- фон
MEAN = np.array([0.485, 0.456, 0.406], np.float32)   # нормализация ImageNet, как в KaliCalib
STD = np.array([0.229, 0.224, 0.225], np.float32)


def grid_court_m(length_m=28.0, width_m=15.0):
    """Сетка ключевых точек KaliCalib (data/datasets/viewds.py: getFieldPoints) в наших
    осях, метры. У KaliCalib: сантиметры, 13 столбцов по длине, 7 рядов по ширине,
    Y_kc = 1500 - s. Их Y растёт к камере, наш -- от камеры, поэтому Y = s
    (проверено по ручной калибровке на images/source/frame.jpg)."""
    xs = np.arange(13) * length_m / 12
    s, u, ys = 0.0, 1.75, []
    for _ in range(7):          # от ближней боковой: 0, 1.75, 3.80, 6.15, 8.80, 11.75, 15.0
        ys.append(s * width_m / 15.0)
        s, u = s + u, u + 0.30
    return np.array([[x, y] for y in ys for x in xs])


class KaliCalib:
    def __init__(self, weights, device=None):
        import torch   # torch нужен только здесь: без него остальной пакет работает
        from basketball.kalicalib.model import makeModel

        if device is None:
            device = ("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
        self.torch, self.device = torch, torch.device(device)
        self.model = makeModel()
        # weights_only=True: не исполнять произвольный pickle-код из скачанного файла
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.model.to(self.device).eval()

    def heatmaps(self, frame_bgr):
        """(94, 135, 240) вероятностей; кадр любого размера растягивается до 960x540,
        как при обучении (пропорции не сохраняются)."""
        img = cv2.cvtColor(cv2.resize(frame_bgr, (NET_W, NET_H), interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = self.torch.from_numpy(((img - MEAN) / STD).transpose(2, 0, 1)).unsqueeze(0)
        with self.torch.no_grad():
            return self.model(x.to(self.device))[0].cpu().numpy()

    def keypoints(self, frame_bgr, length_m=28.0, width_m=15.0):
        """-> (image_px (N, 2), court_m (N, 2), scores (N,)) только для найденных точек.
        Как в KaliCalib: пиксель отдаётся каналу с максимальной вероятностью, точка --
        центр масс своей области; score -- пик вероятности в ней."""
        hm = self.heatmaps(frame_bgr)
        owner = hm.argmax(axis=0)
        h, w = frame_bgr.shape[:2]
        scale = np.array([w / NET_W, h / NET_H]) * STRIDE
        grid = grid_court_m(length_m, width_m)
        img, crt, scores = [], [], []
        for i in range(N_GRID):
            region = (owner == i).astype(np.uint8)
            m = cv2.moments(region, binaryImage=True)
            if m["m00"] == 0:
                continue
            img.append(np.array([m["m10"], m["m01"]]) / m["m00"] * scale)
            crt.append(grid[i])
            scores.append(float(hm[i][region > 0].max()))
        return np.array(img).reshape(-1, 2), np.array(crt).reshape(-1, 2), np.array(scores)

    def homography(self, frame_bgr, min_score=0.0, length_m=28.0, width_m=15.0, thresh_px=None):
        """CourtHomography по точкам сети. Порог RANSAC по умолчанию -- 35 px при
        ширине 960, как в KaliCalib, пересчитанный под размер кадра."""
        img, crt, scores = self.keypoints(frame_bgr, length_m, width_m)
        keep = scores >= min_score
        if thresh_px is None:
            thresh_px = 35.0 * frame_bgr.shape[1] / NET_W
        names = [f"kp{i}" for i in range(int(keep.sum()))]
        return CourtHomography.from_keypoints(img[keep], crt[keep], length_m, width_m,
                                              thresh_px=thresh_px, names=names)
