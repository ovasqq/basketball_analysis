"""Обработка видео: чтение кадров, выпрямление объектива, отрисовка кадра с мини-картой.
source.py -- FrameSource, lens.py -- Lens, render.py -- Renderer. Точка входа -- video.py."""
from basketball.video.lens import Lens
from basketball.video.render import Renderer
from basketball.video.source import FrameSource

__all__ = ["FrameSource", "Lens", "Renderer"]
