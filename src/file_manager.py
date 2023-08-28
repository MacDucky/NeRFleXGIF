import os
import json
from typing import Literal

import numpy as np
from pathlib import Path
from PIL import Image

from nerfstudio.data.dataparsers.nerfstudio_dataparser import Nerfstudio, NerfstudioDataParserConfig
from src.data_classes import Transforms, dict_to_transforms, CameraPath, Keyframe, CameraPose


class FileManager:
    BASE_DATA_PATH = Path('data/nerfstudio')
    BASE_OUTPUTS = Path('outputs')

    def __init__(self, path_to_config: str | os.PathLike | Path):
        self._config_path = Path(path_to_config)

        self._data_name = self._config_path.relative_to('/workspace/outputs/').parents[-2].name
        self._path_to_data = self.BASE_DATA_PATH.joinpath(self._data_name)

        self._train_name = self._config_path.parent.name
        self._path_to_output = self.BASE_OUTPUTS.joinpath(f'{self._data_name}/nerfacto/{self._train_name}')

        self._cam_path_dir = self._path_to_data.joinpath('camera_paths')
        self._transforms_path = self._path_to_data.joinpath('transforms.json')
        self._transforms = None

        self._data_parser: Nerfstudio = NerfstudioDataParserConfig(
            data=self._path_to_data,
            # downscale_factor=8,
            # auto_scale_poses=False,
        ).setup()
        self._dp_outputs = self._data_parser.get_dataparser_outputs('train')
        self._viewer_poses = None

    def load_transforms_file(self) -> Transforms:
        if self._transforms is None:
            with open(self._transforms_path, 'r') as f:
                self._transforms = json.load(f, object_hook=dict_to_transforms)
        return self._transforms

    @property
    def viewer_poses(self):
        if self._viewer_poses is None:
            self._viewer_poses = self._dp_outputs.cameras.camera_to_worlds.numpy()
            tile = np.tile(np.array([0, 0, 0, 1]), (self._viewer_poses.shape[0], 1))
            self._viewer_poses = np.concatenate((self._viewer_poses, tile.reshape((-1, 1, 4))), axis=1)
        return self._viewer_poses

    def get_original_positions(self, camera_convention: Literal['opencv', 'opengl']):
        return self._dp_outputs.transform_poses_to_original_space(self._dp_outputs.cameras.camera_to_worlds,
                                                                  camera_convention)

    def generate_cam_path_file(self, filename: str, start_idx: int, end_idx: int, fov: int, look_at_cameras):
        if not filename.lower().endswith('json'):
            filename = f'{filename}.json'
        first_image_path = next(self._path_to_data.joinpath('images').iterdir())
        image = Image.open(first_image_path)
        width, height = image.size
        start_keyframe = Keyframe(self.viewer_poses[start_idx - 1], fov)
        end_keyframe = Keyframe(self.viewer_poses[end_idx - 1], fov)
        keyframes = [start_keyframe, end_keyframe]
        fps = 24
        seconds = 2
        assert len(look_at_cameras) == fps * seconds, \
            f'Transition frames should be of length {fps}*{seconds}={fps * seconds}'
        cam_path = CameraPath(keyframes, 'perspective', height, width,
                              [CameraPose(lookat, fov) for lookat in look_at_cameras],
                              fps, seconds)
        with open(self._cam_path_dir.joinpath(filename), 'w') as fp:
            json.dump(cam_path.to_dict(), fp)


if __name__ == '__main__':
    x = FileManager('/workspace/outputs/poster/nerfacto/2023-08-18_164728/config.yml')
    x.generate_cam_path_file('test_class', 10, 25, 50, np.zeros(shape=(5, 4, 4)))
    i = 0
