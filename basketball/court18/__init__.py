"""YOLOv8-pose на 18 точек корта (HoopVision / HanaFEKI / nba-vision).
court18.py -- шаблон точек и обёртка кадр -> CourtHomography; нужен ultralytics."""
from basketball.court18.court18 import Court18, template_court_m

__all__ = ["Court18", "template_court_m"]
