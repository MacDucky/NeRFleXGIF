import os
import cv2
from pathlib import Path


class VideoData:
    def __init__(self, path_to_vid: str | os.PathLike | Path):
        self._path = path_to_vid
        self._cap = cv2.VideoCapture(str(path_to_vid))
        self._fps = None
        self._frame_count = None
        self._height = None
        self._width = None

    @property
    def fps(self) -> float:
        if self._fps is None:
            self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        return self._fps

    @property
    def frame_count(self) -> int:
        if self._frame_count is None:
            self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return self._frame_count

    @property
    def width(self) -> int:
        if self._width is None:
            self._width = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        return int(self._width)

    @property
    def height(self) -> int:
        if self._height is None:
            self._height = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return int(self._height)
