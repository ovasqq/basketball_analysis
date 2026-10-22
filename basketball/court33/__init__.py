"""Roboflow basketball-court-detection-2 на 33 точки (roboflow/sports, cs25m116, nba_3d).
court33.py -- шаблон точек и обёртка кадр -> CourtHomography; нужен пакет inference."""
from basketball.court33.court33 import Court33, template_court_m

__all__ = ["Court33", "template_court_m"]
