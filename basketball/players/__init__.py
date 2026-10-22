"""Детекция игроков (YOLO, COCO person) и пиксель «низа ног» для перевода на корт.
players.py -- обёртка кадр -> Players(boxes, scores, feet, ids); нужен ultralytics."""
from basketball.players.players import PlayerDetector, Players, feet_from_boxes

__all__ = ["PlayerDetector", "Players", "feet_from_boxes"]
