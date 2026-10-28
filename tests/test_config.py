"""Проверки параметров CLI из JSON-конфига. Запуск из корня проекта: python -m tests.test_config"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from basketball.config import parse_args
from basketball.paths import ROOT


def _parser():
    ap = argparse.ArgumentParser(prog="t")
    ap.add_argument("video", nargs="?")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--lens")
    ap.add_argument("--px-per-m", type=float, default=40.0)
    return ap


def _run(cfg, argv=()):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.json"
        p.write_text(json.dumps(cfg))
        return parse_args(_parser(), ["-c", str(p), *argv], required=("video",),
                          paths=("video", "lens"))


def test_config_values_and_cli_override():
    a = _run({"video": "v.mp4", "frames": 10, "px_per_m": 30, "_comment": "x"})
    assert a.frames == 10 and a.px_per_m == 30
    assert a.video == str(ROOT / "v.mp4")                     # путь -- от корня проекта
    a = _run({"video": "v.mp4", "frames": 10}, ["--frames", "5", "other.mp4"])
    assert a.frames == 5 and a.video == "other.mp4"           # флаги CLI важнее конфига


def test_path_with_camera_suffix():
    a = _run({"video": "/abs/v.mp4", "lens": "data/cal.json:ace_4"})
    assert a.video == "/abs/v.mp4" and a.lens == str(ROOT / "data/cal.json") + ":ace_4"


def test_unknown_key_and_missing_required_fail():
    for cfg in ({"video": "v.mp4", "frams": 10}, {"frames": 10}):
        try:
            _run(cfg)
        except SystemExit:
            continue
        raise AssertionError(f"конфиг {cfg} должен был упасть")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
