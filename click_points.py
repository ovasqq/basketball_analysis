"""Маленький UI: кликаете по фото, координаты (в пикселях ОРИГИНАЛА) печатаются в консоль.

    python click_points.py images/source/frame.jpg
    python click_points.py images/source/frame.jpg --scale 1      # показать без уменьшения

Управление:
    левый клик   -- добавить точку
    правый клик или u -- убрать последнюю
    q / Esc / закрыть окно -- выйти и напечатать JSON-заготовку для calibration.json

Если картинка не помещается на экран, она показывается уменьшенной, а клики
пересчитываются в оригинальные координаты. Цена: точность клика равна
1/scale пикселя оригинала (при scale 0.5 -- 2 px). Для точной разметки углов
используйте --scale 1 или кадрируйте.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2


class ClickCollector:
    def __init__(self, image, max_w=1400, max_h=850, scale=None):
        self.image = image
        h, w = image.shape[:2]
        self.scale = float(scale) if scale else min(1.0, max_w / w, max_h / h)
        self.display = (image.copy() if self.scale == 1.0 else
                        cv2.resize(image, None, fx=self.scale, fy=self.scale,
                                   interpolation=cv2.INTER_AREA))
        self.points = []          # в координатах оригинала, (x, y)

    def on_mouse(self, event, x, y, flags=0, param=None):
        if event == cv2.EVENT_LBUTTONDOWN:
            px, py = x / self.scale, y / self.scale
            self.points.append((px, py))
            print(f"#{len(self.points)}: x={px:.1f}, y={py:.1f}")
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.undo()

    def undo(self):
        if self.points:
            x, y = self.points.pop()
            print(f"убрана точка ({x:.1f}, {y:.1f}); осталось {len(self.points)}")

    def render(self):
        img = self.display.copy()
        for i, (px, py) in enumerate(self.points, 1):
            c = (round(px * self.scale), round(py * self.scale))
            cv2.circle(img, c, 5, (0, 0, 255), -1)
            cv2.circle(img, c, 7, (255, 255, 255), 1)
            cv2.putText(img, str(i), (c[0] + 9, c[1] - 9), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 3)
            cv2.putText(img, str(i), (c[0] + 9, c[1] - 9), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 1)
        return img

    def json_snippet(self, source):
        """Готовый calibration.json: размер кадра записан сразу, image_rel -- клик в
        долях ширины/высоты. court_m намеренно оставлен как FILL_X/FILL_Y: JSON с ними
        не загрузится, пока вы не впишете метры, и забыть это сделать не получится."""
        h, w = self.image.shape[:2]
        pts = ",\n".join(f'    {{"name": "p{i}", "image_rel": [{x / w:.5f}, {y / h:.5f}], '
                         f'"court_m": [FILL_X, FILL_Y], "role": "fit"}}'
                         for i, (x, y) in enumerate(self.points, 1))
        return ("{\n"
                '  "court": {"length_m": 28.0, "width_m": 15.0},\n'
                f'  "image": {{"source": {json.dumps(source, ensure_ascii=False)}, '
                f'"width_px": {w}, "height_px": {h}}},\n'
                f'  "points": [\n{pts}\n  ]\n}}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--scale", type=float, default=None,
                    help="масштаб показа (по умолчанию: уменьшить до экрана, если нужно)")
    a = ap.parse_args()

    img = cv2.imread(a.image)
    if img is None:
        sys.exit(f"не удалось прочитать изображение: {a.image}")
    h, w = img.shape[:2]
    cc = ClickCollector(img, scale=a.scale)
    print(f"изображение {w}x{h}, масштаб показа {cc.scale:.3f}. "
          "Клик -- точка, правый клик/u -- отмена, q/Esc -- выход.")

    win = "click_points"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, cc.on_mouse)
    while True:
        cv2.imshow(win, cc.render())
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("u"):
            cc.undo()
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()

    if cc.points:
        print("\nЗаготовка calibration.json (впишите court_m в метрах вместо FILL_*, "
              "проверьте размеры корта):")
        print(cc.json_snippet(os.path.basename(a.image)))


if __name__ == "__main__":
    main()