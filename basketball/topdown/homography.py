"""Гомография изображение <-> корт (метры) <-> вид сверху. Оценка через OpenCV.

Минимум: 4 пары точек. Больше точек -> OpenCV (RANSAC + уточнение по инлайерам)
строит лучшую по набору гомографию и отбрасывает выбросы (опечатки в кликах).

CLI: см. main.py.

Из кода:
    ch = CourtHomography.from_json("calibration.json")            # размер кадра из JSON
    ch = CourtHomography.from_json("calibration.json", (1280, 720))  # или под свой кадр
    ch.image_to_court([655, 447])        # -> метры на корте
    ch.image_to_topdown([655, 447])      # -> пиксель на виде сверху
    cv2.imwrite("topdown.png", ch.warp_topdown(frame))
"""
from __future__ import annotations

import json
import warnings
from itertools import combinations

import cv2
import numpy as np


def _pts(a, name):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError(f"{name}: ожидается массив (N, 2), получено {a.shape}")
    return a


def _check_no_collinear_triple(pts, name, min_rel_area=0.02):
    """При ровно 4 точках OpenCV на вырожденной конфигурации (три точки на одной
    прямой) молча возвращает мусорную H. Проверяем сами: площадь любого треугольника
    из точек, в единицах (среднеквадратичный радиус облака)^2, не должна быть ~0."""
    c = pts - pts.mean(axis=0)
    q = c / np.sqrt((c ** 2).sum(axis=1).mean())
    for i, j, k in combinations(range(len(q)), 3):
        a, b = q[j] - q[i], q[k] - q[i]
        if abs(a[0] * b[1] - a[1] * b[0]) / 2 < min_rel_area:
            raise ValueError(f"{name}: точки {i}, {j}, {k} почти лежат на одной прямой "
                             "-- гомография не определена однозначно")


def _fix_sign(H, ref_xy):
    """H определена с точностью до масштаба и знака. Фиксируем: ||H||=1 и w>0 в ref_xy."""
    H = H / np.linalg.norm(H)
    w = H[2] @ np.array([ref_xy[0], ref_xy[1], 1.0])
    return -H if w < 0 else H


def _apply(H, pts):
    """Применить H к (2,) или (M, 2). Точки за линией горизонта (w <= 0) -> NaN.
    (cv2.perspectiveTransform в этом случае молча возвращает мусор.)"""
    p = np.asarray(pts, dtype=np.float64)
    single = p.ndim == 1
    p = np.atleast_2d(p)
    ph = np.column_stack([p, np.ones(len(p))]) @ H.T
    w = ph[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = ph[:, :2] / w[:, None]
    out[w <= 0] = np.nan
    return out[0] if single else out


class CourtHomography:
    """Конвенция осей корта: X вдоль длины (0..length_m), Y поперёк (0..width_m),
    Y растёт от камеры к дальней боковой линии. В виде сверху дальняя сторона
    находится вверху (как на кадре), масштаб -- px_per_m пикселей на метр."""

    def __init__(self, image_pts, court_pts, length_m=28.0, width_m=15.0,
                 thresh_px=3.0, method=cv2.RANSAC, names=None):
        img = _pts(image_pts, "image_pts")
        crt = _pts(court_pts, "court_pts")
        if len(img) != len(crt):
            raise ValueError(f"разное число точек: {len(img)} и {len(crt)}")
        if len(img) < 4:
            raise ValueError(f"нужно минимум 4 пары точек, получено {len(img)}")

        if len(img) == 4:   # ровно 4 точки: OpenCV не проверяет вырожденность
            _check_no_collinear_triple(img, "image_pts")
            _check_no_collinear_triple(crt, "court_pts")

        # Оцениваем корт -> изображение: шум кликов живёт в пикселях, поэтому порог
        # RANSAC (thresh_px) осмыслен именно там. Обратную H получаем инверсией.
        # При ровно 4 точках OpenCV считает точное решение, RANSAC не нужен.
        H_c2i, mask = cv2.findHomography(crt, img, method, thresh_px)
        if H_c2i is None:
            raise ValueError("OpenCV не смог оценить гомографию "
                             "(вырожденная конфигурация: три точки почти на одной прямой?)")
        # method=0 (обычные наименьшие квадраты) маску инлайеров не возвращает
        # (или возвращает пустую) -- тогда считаем все точки используемыми.
        self.inliers = (np.ones(len(img), bool) if mask is None or not mask.any()
                        else mask.ravel().astype(bool))
        ref_c = crt[self.inliers].mean(axis=0)
        ref_i = img[self.inliers].mean(axis=0)
        self.H_court2img = _fix_sign(H_c2i, ref_c)
        self.H_img2court = _fix_sign(np.linalg.inv(self.H_court2img), ref_i)

        self.image_pts, self.court_pts = img, crt
        self.length_m, self.width_m = float(length_m), float(width_m)
        self.names = names or [f"p{i}" for i in range(len(img))]

    @classmethod
    def from_json(cls, path, image_size=None, **kw):
        """Читает формат calibration.json (поле role игнорируется: берутся все точки).

        image_rel -- клик в долях ширины/высоты кадра (0..1), court_m -- точка корта
        в метрах. image_size=(w, h) -- размер кадра в пикселях, к которому применяется
        гомография; по умолчанию image.width_px/height_px из JSON. Так одна калибровка
        подходит к любому разрешению того же ракурса -- но только при том же
        соотношении сторон: обрезка или чёрные полосы сдвигают координаты."""
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        pts = d["points"]
        ref = (d["image"]["width_px"], d["image"]["height_px"])
        if image_size is None:
            image_size = ref
        elif abs(image_size[0] / image_size[1] / (ref[0] / ref[1]) - 1) > 0.01:
            warnings.warn(f"соотношение сторон кадра {image_size[0]}x{image_size[1]} не совпадает "
                          f"с калибровочным {ref[0]}x{ref[1]}: кадр обрезан или с полями? "
                          "Гомография будет неверной.", stacklevel=2)
        img = _pts([p["image_rel"] for p in pts], "image_rel") * np.asarray(image_size, float)
        return cls(img, [p["court_m"] for p in pts],
                   d["court"]["length_m"], d["court"]["width_m"],
                   names=[p.get("name") for p in pts], **kw)

    @classmethod
    def from_matrix(cls, H_img2court, length_m=28.0, width_m=15.0):
        """Готовая H (кадр -> метры корта), например из <позиции>.meta.json video.py.
        Точки для конструктора -- сетка 5x5 по корту, спроецированная этой H (точки за
        горизонтом отбрасываются), поэтому результат совпадает с исходной H."""
        Hc2i = np.linalg.inv(np.asarray(H_img2court, np.float64))
        crt = np.array([[x, y] for x in np.linspace(0, length_m, 5)
                        for y in np.linspace(0, width_m, 5)])
        img = _apply(Hc2i, crt)
        ok = np.isfinite(img).all(axis=1)
        return cls(img[ok], crt[ok], length_m, width_m, method=0)

    @classmethod
    def from_keypoints(cls, image_pts, court_pts, length_m=28.0, width_m=15.0, orient=True, **kw):
        """Гомография по точкам нейросети. Шаблоны сетей из разных проектов отсчитывают
        оси корта по-своему (Y от дальней боковой, X справа налево и т.п.), поэтому
        при orient=True оси приводятся к нашей конвенции по самому кадру: в центре
        корта +X должен идти вправо по кадру, +Y -- вверх (от камеры). Если нет,
        точки шаблона отражаются (x -> L - x и/или y -> W - y) и H считается заново."""
        crt = _pts(court_pts, "court_pts").copy()
        ch = cls(image_pts, crt, length_m, width_m, **kw)
        if not orient:
            return ch
        c = np.array([length_m / 2, width_m / 2])
        p0, px, py = ch.court_to_image([c, c + [1.0, 0.0], c + [0.0, 1.0]])
        if not np.isfinite([p0, px, py]).all():
            return ch           # центр корта за горизонтом: направление осей не определить
        flip_x, flip_y = px[0] < p0[0], py[1] > p0[1]
        if not (flip_x or flip_y):
            return ch
        if flip_x:
            crt[:, 0] = length_m - crt[:, 0]
        if flip_y:
            crt[:, 1] = width_m - crt[:, 1]
        return cls(image_pts, crt, length_m, width_m, **kw)

    # --------------------------------------------------- изображение <-> корт (метры)
    def image_to_court(self, pts):
        return _apply(self.H_img2court, pts)

    def court_to_image(self, pts):
        return _apply(self.H_court2img, pts)

    # --------------------------------------------------- изображение <-> вид сверху (px)
    def _S(self, px_per_m):
        s = float(px_per_m)  # метры корта -> пиксели вида сверху, Y перевёрнут (дальняя сторона вверху)
        return np.array([[s, 0.0, 0.0], [0.0, -s, self.width_m * s], [0.0, 0.0, 1.0]])

    def image_to_topdown(self, pts, px_per_m=40.0):
        return _apply(self._S(px_per_m) @ self.H_img2court, pts)

    def topdown_to_image(self, pts, px_per_m=40.0):
        return _apply(self.H_court2img @ np.linalg.inv(self._S(px_per_m)), pts)

    def warp_topdown(self, frame, px_per_m=40.0):
        """Кадр -> изображение вида сверху размером (length_m*s) x (width_m*s)."""
        size = (round(self.length_m * px_per_m), round(self.width_m * px_per_m))
        return cv2.warpPerspective(frame, self._S(px_per_m) @ self.H_img2court, size)

    # --------------------------------------------------- диагностика
    def errors(self):
        """(err_px, err_m) по каждой точке: расхождение с кликом в пикселях (корт->кадр)
        и в метрах (кадр->корт). При ровно 4 точках ошибки нулевые по построению."""
        err_px = np.linalg.norm(self.court_to_image(self.court_pts) - self.image_pts, axis=1)
        err_m = np.linalg.norm(self.image_to_court(self.image_pts) - self.court_pts, axis=1)
        return err_px, err_m

    def report(self):
        err_px, err_m = self.errors()
        rows = [f"точек: {len(self.image_pts)}, инлайеров: {int(self.inliers.sum())}",
                f"{'точка':<26}{'инлайер':<9}{'ошибка px':>10}{'ошибка м':>10}"]
        for n, ok, ep, em in zip(self.names, self.inliers, err_px, err_m):
            rows.append(f"{n:<26}{'да' if ok else 'НЕТ':<9}{ep:>10.2f}{em:>10.3f}")
        return "\n".join(rows)
