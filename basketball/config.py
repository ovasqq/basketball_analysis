"""Параметры CLI из JSON-конфига: `-c configs/xxx.json`, флаги командной строки важнее.

Ключи конфига -- имена флагов (dest argparse): `frames` = `--frames`,
`px_per_m` = `--px-per-m`, позиционные -- по своему имени (`video`, `images`).
Неизвестный ключ -- ошибка (опечатка не должна молча игнорироваться).
Относительные пути в конфиге считаются от корня проекта, а не от текущей папки;
для значений вида «путь:камера» (как --lens) -- путь до двоеточия.

Из кода (вместо ap.parse_args()):
    a = parse_args(ap, required=("video",), paths=("video", "lens", "out"))
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from basketball.paths import ROOT


def _resolve(value, root):
    if not isinstance(value, str) or not value:
        return value
    path, sep, rest = value.partition(":")
    p = Path(path)
    return value if p.is_absolute() else str(root / p) + sep + rest


def load_config(path, ap, paths=()):
    """JSON -> dict для ap.set_defaults; проверяет ключи по флагам парсера."""
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise SystemExit(f"{path}: ожидается JSON-объект {{\"ключ\": значение}}")
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}     # _comment и т.п.
    known = {a.dest for a in ap._actions} - {"help", "config"}
    unknown = sorted(set(cfg) - known)
    if unknown:
        raise SystemExit(f"{path}: неизвестные ключи {unknown}; допустимые: {sorted(known)}")
    for k in paths:
        if k in cfg:
            cfg[k] = ([_resolve(v, ROOT) for v in cfg[k]] if isinstance(cfg[k], list)
                      else _resolve(cfg[k], ROOT))
    return cfg


def parse_args(ap, argv=None, required=(), paths=()):
    """ap.parse_args() с поддержкой -c/--config. Позиционные аргументы, которые
    могут прийти из конфига, объявляйте с nargs="?" (или "*") и перечисляйте в
    required: их наличие проверяется после слияния конфига и флагов."""
    ap.add_argument("-c", "--config", metavar="JSON",
                    help="конфиг с параметрами (ключи = имена флагов); флаги CLI его переопределяют")
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("-c", "--config")
    known, _ = pre.parse_known_args(argv)
    if known.config:
        ap.set_defaults(**load_config(known.config, ap, paths))
    a = ap.parse_args(argv)
    missing = [k for k in required if getattr(a, k) in (None, [])]
    if missing:
        ap.error(f"не задано: {', '.join(missing)} (в командной строке или в конфиге)")
    return a
