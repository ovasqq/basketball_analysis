"""KaliCalib (CEA-LIST, MMSports 2022): 91 точка сетки по полу корта.
model.py -- архитектура сети (копия), kalicalib.py -- обёртка кадр -> CourtHomography.
Импорт пакета torch не требует: он нужен только при создании KaliCalib."""
from basketball.kalicalib.kalicalib import KaliCalib, grid_court_m

__all__ = ["KaliCalib", "grid_court_m"]
