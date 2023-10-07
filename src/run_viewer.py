import os
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms

from multiprocessing import Process, set_start_method
from nerfstudio.scripts.viewer.run_viewer import RunViewer
from nerfstudio.viewer.viser.server import ViserServer


def start_viewer(config_path: Path | os.PathLike | str, background: bool = True) -> None | Process:
    """
    Starts the web viewer of a trained model.
    :param config_path: configuration path to **config.yml**
    :param background: If true then returns the Process object to be terminated manually later on. Otherwise,
        termination is based on a busy wait.
    :return:
    """
    path_to_config = Path(config_path)
    stop_viewer_file = Path('/tmp/stop_viewer')
    if not path_to_config.exists():
        print(f'{" Path to config file does not exist, exiting!! ":=^70}')
        sys.exit(-1)
    viewer = RunViewer(path_to_config)
    set_start_method('spawn')
    viewer_proc = Process(target=viewer.main, name='nerf_viewer')
    viewer_proc.start()
    if background:
        return viewer_proc
    print(f'\nViewer Started. To close the viewer enter the command:\n'
          f'\'docker exec {os.environ.get("HOSTNAME")} touch {stop_viewer_file}\' in command line.\n')
    while not stop_viewer_file.exists():
        time.sleep(3)
    print(f'{" stop_viewer file detected ":=^70}')
    print(f'{" Terminating viewer! ":=^70}')
    viewer_proc.terminate()
    os.remove(stop_viewer_file)
    print('Done!')


class ImageAdder:
    def __init__(self):
        self.viser_server = ViserServer('0.0.0.0', 7007)
        self.next_image_index = 0



    def add_image(self, cameras):
        camera_json = cameras.to_json(self.next_image_index, self.create_number_image(100, 50, self.next_image_index),
                                      max_size=100)
        # self.viser_server.add_dataset_image(idx=f'{self.next_image_index:06d}', json=camera_json)


if __name__ == '__main__':
    # from nerfstudio.data.dataparsers.nerfstudio_dataparser import Nerfstudio, NerfstudioDataParserConfig
    # from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
    # from transform_inputs import path_to_data
    # from nerfstudio.cameras.camera_paths import get_interpolated_camera_path

    start_viewer(r'/workspace/outputs/chair/nerfacto/2023-09-02_163102/config.yml', background=False)
    # adder = ImageAdder()
    # parser: Nerfstudio = NerfstudioDataParserConfig(
    #     data=path_to_data,
    #     downscale_factor=8,
    # ).setup()
    # x: DataparserOutputs = parser.get_dataparser_outputs(split='train')
    # # adder.add_image()
