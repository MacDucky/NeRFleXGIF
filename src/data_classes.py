import os
import re
import shutil
import numpy as np
from pathlib import Path
from functools import total_ordering
from typing import Literal
from dataclasses import dataclass, field, fields, InitVar, asdict
from contextlib import contextmanager


@contextmanager
def temporary_file_change(file_path: str | os.PathLike | Path):
    current_file = Path(file_path)
    real_file = current_file.parent.joinpath('real_file')
    shutil.copy(current_file, real_file)
    yield
    shutil.move(real_file, current_file)


@dataclass
class CameraIntrinsics:
    fl_x: float
    fl_y: float
    k1: float
    k2: float
    p1: float
    p2: float
    cx: float
    cy: float
    w: int
    h: int
    applied_transform: list[list[float]] = field(default_factory=lambda: np.identity(3).tolist())
    aabb_scale: int = field(default=1)

    # todo: add camera_model == opencv somewhere

    def to_dict(self):
        return asdict(self)


@total_ordering
class Frame:
    INDEX_FINDER = re.compile(r'.*_(\d+)\..*')

    def __init__(self, file_path: str, transform_matrix: np.ndarray, set_colmap_id: bool = False):
        self.file_path = file_path
        self.transform_matrix = transform_matrix
        if set_colmap_id:
            self.colmap_im_id = int(self.INDEX_FINDER.search(self.file_path).group(1))
        else:
            self.colmap_im_id = None

    def __lt__(self, other):
        my_index = int(self.INDEX_FINDER.search(self.file_path).group(1))
        other_index = int(self.INDEX_FINDER.search(other.file_path).group(1))
        return my_index < other_index

    def __eq__(self, other):
        return not (self < other or self > other)

    def to_dict(self):
        dct = {}
        dct['file_path'] = self.file_path
        dct['transform_matrix'] = self.transform_matrix.tolist()
        if self.colmap_im_id is not None:
            dct['colmap_im_id'] = self.colmap_im_id
        return dct


@dataclass
class Transforms:
    intrinsics: CameraIntrinsics
    frames: list[Frame]

    def to_dict(self, past_train: bool = False):
        dct = {}
        dct.update(self.intrinsics.to_dict())
        if past_train:  # do this for full cameras matrices instead of just train.
            dct['train_filenames'] = [frame.file_path for frame in self.frames]
        dct['frames'] = [frame.to_dict() for frame in self.frames]
        return dct


def dict_to_transforms(dct: dict):
    d_keys = set(dct.keys())
    frame_fields = {'file_path', 'transform_matrix'}
    if len(frame_fields.intersection(d_keys)) > 0:
        f_path = dct.get('file_path')
        t_mat = dct.get('transform_matrix')
        return Frame(f_path, np.array(t_mat), 'colmap_im_id' in d_keys)

    intrinsics_keys = d_keys.intersection([x.name for x in fields(CameraIntrinsics)])
    sub_dict = {k: v for k, v in dct.items() if k in intrinsics_keys}
    intrinsics: CameraIntrinsics = CameraIntrinsics(**sub_dict)

    frames: list[Frame] = dct.get('frames')
    frames.sort()
    return Transforms(intrinsics, frames)


class KeyFrameAttributes:
    FRAME_NUM = 0
    TIME = 0

    @classmethod
    def get_next_keyframe(cls):
        cls.FRAME_NUM += 1
        return cls.FRAME_NUM

    @classmethod
    def get_time(cls):
        current_time = cls.TIME
        cls.TIME += 1
        assert current_time < 2, 'Keyframe time must be either \'0\' or \'1\'!'
        return current_time


# _aspect: float = 1.5196195005945303
_aspect: float = 1.


@dataclass
class Keyframe:
    np_matrix: InitVar[np.ndarray]
    matrix: str = field(init=False)
    fov: int
    aspect: float = _aspect
    properties: str = field(default='[["FOV",{fov}],["NAME","CAMERA {camera}"], ["TIME",{time}]]', init=False)

    def __post_init__(self, np_matrix):
        mat = np_matrix.flatten()
        self.matrix = np.array2string(mat, separator=',').replace('\n', '').replace(' ', '')
        attribs = KeyFrameAttributes
        self.properties = self.properties.format(fov=self.fov, camera=attribs.get_next_keyframe(),
                                                 time=attribs.get_time())

    def to_dict(self):
        return asdict(self)


@dataclass
class CameraPose:
    look_at_cam: InitVar[np.ndarray]
    camera_to_world: list[float] = field(init=False)
    fov: int
    aspect: float = _aspect

    def __post_init__(self, look_at_cam: np.ndarray):
        self.camera_to_world = look_at_cam.flatten().tolist()

    def to_dict(self):
        return asdict(self)


@dataclass
class CameraPath:
    keyframes: list[Keyframe]
    camera_type: Literal['perspective', 'fisheye', 'equirectangular']
    render_height: int
    render_width: int
    camera_path: list[CameraPose]
    fps: int
    seconds: float
    smoothness_value: int = field(default=0, init=False)
    is_cycle: bool = field(default=False, init=False)
    crop: None = field(default=None, init=False)

    def to_dict(self):
        return asdict(self)
