"""Видео «кадр с разметкой + мини-карта» по результатам video.py, без повторной детекции.

Берёт <позиции>.jsonl и <позиции>.meta.json (рядом): из meta -- исходное видео,
объектив, отрезок кадров и гомографию корта; из jsonl -- игроков на каждом кадре.
Слева кадр (выпрямленный) с разметкой и игроками, справа схема корта с точками и
хвостами траекторий. Игроки с флагами (cut, off_court) серые и на схему не идут.

    python render_video.py images/output/ace_4.jsonl                       # -> images/output/ace_4.mp4
    python render_video.py -c configs/render_ace_4.json
    python render_video.py images/output/ace_4.jsonl --frames 300 --trail 90

Запуск из корня проекта.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from basketball.config import parse_args
from basketball.topdown import CourtHomography
from basketball.video import FrameSource, Lens, Renderer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("positions", nargs="?", help="jsonl от video.py (meta.json -- рядом)")
    ap.add_argument("--out", help="куда сохранить видео (по умолчанию <позиции>.mp4)")
    ap.add_argument("--frames", type=int, default=None,
                    help="отрисовать только первые N записей jsonl")
    ap.add_argument("--px-per-m", type=float, default=30.0, help="масштаб мини-карты до подгонки по высоте")
    ap.add_argument("--trail", type=int, default=45, help="длина хвоста траектории, кадров")
    a = parse_args(ap, required=("positions",), paths=("positions", "out"))

    positions = Path(a.positions)
    meta_path = positions.with_suffix(".meta.json")
    if not meta_path.is_file():
        raise SystemExit(f"нет {meta_path}: render_video.py работает по результатам video.py")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    args = meta["args"]
    with positions.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    if a.frames is not None:
        records = records[:a.frames]
    if not records:
        raise SystemExit(f"{positions}: пусто")

    ch = CourtHomography.from_matrix(meta["H_img2court"], *meta["court_m"])
    first, last = records[0]["frame"], records[-1]["frame"]
    source = FrameSource(args["video"], start=first, count=last - first + 1, stride=args["stride"])
    lens = None
    if args.get("lens"):
        path, _, camera = args["lens"].partition(":")
        lens = Lens.from_json(path, camera or None, image_size=(source.width, source.height))

    by_frame = {r["frame"]: r for r in records}
    renderer = Renderer(ch, px_per_m=a.px_per_m, trail=a.trail)
    out = Path(a.out or positions.with_suffix(".mp4"))
    out.parent.mkdir(parents=True, exist_ok=True)
    writer, n, t0 = None, 0, time.perf_counter()
    for idx, _, frame in source:
        rec = by_frame.get(idx)
        if rec is None:
            continue
        if lens:
            frame = lens.undistort(frame)
        img = renderer.draw(frame, rec)
        if writer is None:
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                                     (meta.get("fps") or 30.0) / args["stride"],
                                     (img.shape[1], img.shape[0]))
        writer.write(img)
        n += 1
    if writer:
        writer.release()
    if n < len(records):
        print(f"внимание: отрисовано {n} из {len(records)} кадров (видео короче jsonl?)")
    print(f"готово: {n} кадров за {time.perf_counter() - t0:.0f} с -> {out}")


if __name__ == "__main__":
    main()
