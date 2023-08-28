import re
import numpy as np
from functools import total_ordering
from typing import Literal
from dataclasses import dataclass, field, fields, InitVar, asdict


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
    aabb_scale: int


@total_ordering
class Frame:

    def __init__(self, file_path: str, transform_matrix: np.ndarray):
        self.file_path = file_path
        self.transform_matrix = transform_matrix

    def __lt__(self, other):
        index_finder = re.compile(r'.*_(\d+)\..*')
        my_index = int(index_finder.search(self.file_path).group(1))
        other_index = int(index_finder.search(other.file_path).group(1))
        return my_index < other_index

    def __eq__(self, other):
        return not (self < other or self > other)


@dataclass
class Transforms:
    intrinsics: CameraIntrinsics
    frames: list[Frame]


def dict_to_transforms(dct: dict):
    d_keys = set(dct.keys())
    if d_keys == {'file_path', 'transform_matrix'}:
        f_path = dct.get('file_path')
        t_mat = dct.get('transform_matrix')
        return Frame(f_path, np.array(t_mat))

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
