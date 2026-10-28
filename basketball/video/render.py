"""Картинка для видео: слева кадр с разметкой и игроками, справа мини-карта корта.

Рисует по записи кадра -- той же, что пишется в positions.jsonl
({"frame", "t", "players": [{"id", "bbox", "foot_px", "court_m", "flags"}]}),
поэтому видео можно перерисовать по готовому jsonl без повторной детекции.
Игроки с флагами (ноги обрезаны, вне корта) рисуются серым и в хвосты не идут.
"""
from __future__ import annotations

from collections import defaultdict, deque

import cv2
import numpy as np

from basketball.topdown.markings import court_to_minimap, draw_court_topdown, draw_markings

GREY = (150, 150, 150)


def id_color(track_id):
    """Устойчивый цвет трека: оттенок по золотому сечению, -1 (без трека) -- серый."""
    if track_id is None or track_id < 0:
        return GREY
    hue = int((track_id * 0.618034 % 1.0) * 180)
    hsv = np.uint8([[[hue, 220, 230]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


class Renderer:
    def __init__(self, ch, px_per_m=30.0, pad_m=1.0, trail=45):
        self.ch, self.s, self.pad = ch, px_per_m, pad_m
        self.court = draw_court_topdown(ch.length_m, ch.width_m, px_per_m, pad_m)
        self.trails = defaultdict(lambda: deque(maxlen=trail))   # id -> точки мини-карты
        self._markings = None                                      # разметка на кадре: одна на всё видео

    def _frame_layer(self, frame, players):
        if self._markings is None or self._markings[0] != frame.shape:
            mask = draw_markings(self.ch, np.zeros_like(frame), color=(255, 0, 255), alpha=1.0)
            self._markings = (frame.shape, mask.any(axis=2))
        out = frame.copy()
        out[self._markings[1]] = (0.35 * out[self._markings[1]] + 0.65 * np.array((255, 0, 255))
                                  ).astype(np.uint8)
        for p in players:
            color = GREY if p["flags"] else id_color(p["id"])
            x1, y1, x2, y2 = (round(v) for v in p["bbox"])
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            fx, fy = (round(v) for v in p["foot_px"])
            cv2.circle(out, (fx, fy), 5, color, -1)
            if p["id"] >= 0:
                cv2.putText(out, str(p["id"]), (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2, cv2.LINE_AA)
        return out

    def _minimap(self, players):
        img = self.court.copy()
        seen = set()
        for p in players:
            if p["flags"] or p["court_m"] is None:
                continue
            q = court_to_minimap(p["court_m"], self.ch.width_m, self.s, self.pad)[0]
            if p["id"] >= 0:
                self.trails[p["id"]].append(q)
                seen.add(p["id"])
        for tid, pts in self.trails.items():
            if len(pts) > 1:
                cv2.polylines(img, [np.round(np.array(pts)).astype(np.int32)], False,
                              id_color(tid), 2, cv2.LINE_AA)
        for p in players:
            if p["flags"] or p["court_m"] is None:
                continue
            x, y = np.round(court_to_minimap(p["court_m"], self.ch.width_m, self.s, self.pad)[0])
            color = id_color(p["id"])
            cv2.circle(img, (int(x), int(y)), 7, color, -1)
            cv2.circle(img, (int(x), int(y)), 7, (0, 0, 0), 1)
            if p["id"] >= 0:
                cv2.putText(img, str(p["id"]), (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 0), 1, cv2.LINE_AA)
        # Хвосты исчезнувших треков постепенно укорачиваем, чтобы не висели вечно.
        for tid in [t for t in self.trails if t not in seen]:
            self.trails[tid].popleft()
            if not self.trails[tid]:
                del self.trails[tid]
        return img

    def draw(self, frame, record):
        """frame -- выпрямленный кадр, record -- запись кадра (как в positions.jsonl)."""
        left = self._frame_layer(frame, record["players"])
        right = self._minimap(record["players"])
        h = left.shape[0]
        right = cv2.resize(right, (round(right.shape[1] * h / right.shape[0]), h),
                           interpolation=cv2.INTER_AREA)
        out = np.hstack([left, right])
        cv2.putText(out, f"frame {record['frame']}  t={record['t']:.2f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        return out
