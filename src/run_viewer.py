import os
import sys
import time
from pathlib import Path
from nerfstudio.scripts.viewer.run_viewer import RunViewer
from multiprocessing import Process, set_start_method


def start_viewer(config_path: Path | os.PathLike | str, background: bool = True) -> None | Process:
    """
    Starts the web viewer of a trained model
    :param config_path: configuration path to config.yml
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


if __name__ == '__main__':
    start_viewer(r'/workspace/outputs/poster/nerfacto/2023-08-18_164728/config.yml', background=False)
