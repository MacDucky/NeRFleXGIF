import os
import json
import re
import shutil
import numpy as np

from typing import Literal
from pathlib import Path
from PIL import Image

from nerfstudio.data.dataparsers.nerfstudio_dataparser import Nerfstudio, NerfstudioDataParserConfig
from src.data_classes import Transforms, dict_to_transforms, CameraPath, Keyframe, CameraPose, temporary_file_change
from src.gif_maker import create_gif_from_image_dir


class FileManager:
    BASE_DATA_PATH = Path('data/nerfstudio')
    BASE_OUTPUTS = Path('outputs')
    BASE_RENDERS = Path('renders')

    def __init__(self, path_to_config: str | os.PathLike | Path):
        self._config_path = Path(path_to_config)

        self._data_name = self._config_path.relative_to('/workspace/outputs/').parents[-2].name
        self._path_to_data = self.BASE_DATA_PATH.joinpath(self._data_name)

        self._train_name = self._config_path.parent.name
        self._path_to_output = self.BASE_OUTPUTS.joinpath(f'{self._data_name}/nerfacto/{self._train_name}')

        self._cam_path_dir = self._path_to_data.joinpath('camera_paths')
        if not self._cam_path_dir.exists():
            os.makedirs(str(self._cam_path_dir),exist_ok=True)
        self._transforms_path = self._path_to_data.joinpath('transforms.json')
        self._transforms = None

        self._renders_path = self.BASE_RENDERS.joinpath(self._data_name)
        if not self._renders_path.exists():
            os.makedirs(str(self._renders_path), exist_ok=True)
        self._last_rendered_images_path = self._renders_path.joinpath('last_render')
        if not self._last_rendered_images_path.exists():
            os.makedirs(str(self._last_rendered_images_path), exist_ok=True)
        self._data_parser: None | Nerfstudio = None
        self._dp_outputs = None
        self._viewer_poses = None

    @property
    def config_path(self):
        return self._config_path

    @property
    def project_name(self):
        return self._data_name

    @property
    def data_path(self):
        return self._path_to_data

    @property
    def output_path(self):
        return self._path_to_output

    @property
    def cam_dir_path(self):
        return self._cam_path_dir

    @property
    def transforms_path(self):
        return self._transforms_path

    @property
    def renders_dir(self):
        return self._renders_path

    @property
    def last_render_dir(self):
        return self._last_rendered_images_path

    def __merge_real_synthesized_images(self, start_cut_point: int, end_cut_point: int):
        tmp_dir = self._renders_path.joinpath('tmpdir')
        path_to_images = self._path_to_data.joinpath('images')
        images = os.listdir(path_to_images)
        index_or_ext = re.compile(r'.*_(\d+)\.((?:png)|(?:jpe?g))$', re.MULTILINE | re.IGNORECASE)

        # add real images in the range of cut points.
        f_index = 1
        for image in images:
            index = int(index_or_ext.search(image).group(1))
            ext = index_or_ext.search(image).group(2)
            if start_cut_point <= index <= end_cut_point:
                shutil.copy(path_to_images.joinpath(image), tmp_dir.joinpath(f'frame_{f_index:0>5}.{ext}'))
                f_index += 1

        # add synthesized images
        path_to_synthesized = self._last_rendered_images_path
        ext = str(next(path_to_synthesized.iterdir())).split('.')[-1]
        for image in os.listdir(path_to_synthesized):
            shutil.copy(path_to_synthesized.joinpath(image), tmp_dir.joinpath(f'frame_{f_index:0>5}.{ext}'))
            f_index += 1

    def load_transforms_file(self, force_load: bool = False) -> Transforms:
        if self._transforms is None or force_load:
            with open(self._transforms_path, 'r') as f:
                self._transforms = json.load(f, object_hook=dict_to_transforms)
        return self._transforms

    def dump_sorted_transform_file(self, all_transforms: bool = False):
        transform = self.load_transforms_file(force_load=True).to_dict(all_transforms)
        with open(self._transforms_path, 'w') as fp:
            json.dump(transform, fp)

    def viewer_poses(self, all_poses: bool = True, update_poses: bool = False):
        if self._dp_outputs is None:
            update_poses = True

        if update_poses:
            with temporary_file_change(self._transforms_path):
                self.dump_sorted_transform_file(all_poses)
                if self._data_parser is None or self._dp_outputs is None:
                    self._data_parser: Nerfstudio = NerfstudioDataParserConfig(
                        data=self._path_to_data,
                        # downscale_factor=8,
                        # auto_scale_poses=False,
                    ).setup()
                self._dp_outputs = self._data_parser.get_dataparser_outputs('train')

        if self._viewer_poses is None or update_poses:
            self._viewer_poses = self._dp_outputs.cameras.camera_to_worlds.numpy()
            tile = np.tile(np.array([0, 0, 0, 1]), (self._viewer_poses.shape[0], 1))
            self._viewer_poses = np.concatenate((self._viewer_poses, tile.reshape((-1, 1, 4))), axis=1)
        return self._viewer_poses

    def get_original_positions(self, camera_convention: Literal['opencv', 'opengl']):
        return self._dp_outputs.transform_poses_to_original_space(self._dp_outputs.cameras.camera_to_worlds,
                                                                  camera_convention)

    def generate_cam_path_file(self, filename: str, start_idx: int, end_idx: int, fov: int, fps: float,
                               look_at_cameras: list[np.ndarray]):
        if not filename.lower().endswith('json'):
            filename = f'{filename}.json'
        first_image_path = next(self._path_to_data.joinpath('images').iterdir())
        image = Image.open(first_image_path)
        width, height = image.size
        image.close()
        start_keyframe = Keyframe(self.viewer_poses()[start_idx - 1], fov)
        end_keyframe = Keyframe(self.viewer_poses()[end_idx - 1], fov)
        keyframes = [start_keyframe, end_keyframe]
        seconds = (len(look_at_cameras) + 2) / fps
        # assert len(look_at_cameras) == fps * seconds, \
        #     f'Transition frames should be of length {fps}*{seconds}={fps * seconds}'
        cam_path = CameraPath(keyframes, 'perspective', height, width,
                              [CameraPose(lookat, fov) for lookat in look_at_cameras],
                              int(fps), seconds)
        with open(self._cam_path_dir.joinpath(filename), 'w') as fp:
            json.dump(cam_path.to_dict(), fp)
        return self._cam_path_dir.joinpath(filename)

    def create_gif(self, filename: str, cut_points: tuple[int, int], fps):
        # Setup Folders
        if not filename.lower().endswith('.gif'):
            filename += '.gif'

        tmp_path = self._renders_path.joinpath('tmpdir')
        if tmp_path.exists():
            shutil.rmtree(str(tmp_path))
        os.makedirs(name=str(tmp_path), exist_ok=True)

        gifs_folder = tmp_path.parent.joinpath('gifs')
        gifs_folder.mkdir(exist_ok=True)

        # fill up with images....
        self.__merge_real_synthesized_images(*cut_points)
        assert len(os.listdir(tmp_path))

        create_gif_from_image_dir(image_dir=tmp_path, fps=fps, output_path=gifs_folder.joinpath(filename))
        shutil.rmtree(tmp_path, ignore_errors=True)