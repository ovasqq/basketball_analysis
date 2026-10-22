"""Пути проекта. Считаются от расположения этого файла, а не от текущей папки,
поэтому скрипты работают при запуске откуда угодно."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # корень проекта
CALIBRATION = ROOT / "calibration.json"
IMAGES = ROOT / "images"
SOURCE_DIR = IMAGES / "source"     # исходные кадры (в git)
OUTPUT_DIR = IMAGES / "output"     # результаты запусков (в .gitignore)
WEIGHTS_DIR = ROOT / "weights"     # веса нейросетей (в .gitignore, скачиваются отдельно)
KALICALIB_WEIGHTS = WEIGHTS_DIR / "kalicalib_challenge.pth"
COURT18_WEIGHTS = WEIGHTS_DIR / "court18_yolov8_pose.pt"
PLAYERS_WEIGHTS = WEIGHTS_DIR / "yolo11s.pt"      # детектор людей (COCO)
