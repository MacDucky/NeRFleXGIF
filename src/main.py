import math
import os
import re
import shutil
import subprocess
import time

from pathlib import Path
from argparse import ArgumentParser
from typing import Callable

from src.poses import Poses
from src.file_manager import FileManager

from src.video_utils import VideoData


def path(s):
    p = Path(s)
    if not p.exists() or not p.is_file():
        raise TypeError('Path is invalid or is not a file.')
    return p


def get_last_trained_model(trained_dir) -> Path:
    cmd_get_last_file = f'ls -lrt {trained_dir}  | tail -n1 | awk \'{{ print $9 }}\''
    last_trained_config_dir = Path(subprocess.check_output(cmd_get_last_file, shell=True).decode('utf-8').strip())
    return last_trained_config_dir


def execute_and_track_output(cmd: list[str], kill_proc_cond: Callable[[str], bool] = None):
    """
    Execute command and return the output in live.
    :param cmd: Command to execute.
    :param kill_proc_cond: A condition that the process output satisfies that triggers terminate signal(ctrl+c).
    :raises StopIteration: In case on a failure in child process.
    :raises CalledProcessError: In case on a failure in child process.
    :return: A generator object with the subprocess output
    """
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        if kill_proc_cond is not None and kill_proc_cond(stdout_line):
            popen.stdout.close()
            popen.terminate()
            return 'Kill prompt detected, terminated child process.'
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def parsed_args():
    parser = ArgumentParser('NerFlexGif', description='A perfect GIF creator.',
                            epilog="Flow: 1. Process video (COLMAP). 2. Train NeRF model."
                                   "3. Crop excess parts. 4. Synthesize middle frames. 5. Generate GIF!")
    parser.add_argument('-v', '--video-path', type=path, required=True,
                        help="Path to video file to process.")
    parser.add_argument('-p', '--project-name', type=str,
                        help='Name of project. Default is video name.')
    parser.add_argument('--skip-train', action='store_true',
                        help="Skip training, model was trained ahead of time.")
    parser.add_argument('--train-from-checkpoint', action='store_true',
                        help="Continue training from checkpoint")
    parser.add_argument('-g', '--gif-output-name', dest='gif_filename', type=str,
                        help='Name of output GIF. Default is video name.')
    args = parser.parse_args()
    if args.skip_train and args.train_from_checkpoint:
        parser.error('Can not use \'--skip-train\' and \'--train-from-checkpoint\' simultaneously!')
    return parser.parse_args()


if __name__ == '__main__':
    num_frames_target = 300
    # ---------------------------------------  Preprocess video metadata  ----------------------------------------
    # process data - both ns-process && get video metadata
    args = parsed_args()

    video_path: Path = args.video_path
    if args.project_name is not None:
        project_name = args.project_name
    else:
        project_name = video_path.stem

    video_data = VideoData(video_path)

    spacing = video_data.frame_count // num_frames_target
    number_of_frames = math.ceil(video_data.frame_count / spacing) if spacing > 1 else VideoData.frame_count
    fps = (number_of_frames / video_data.frame_count) * video_data.fps
    # ---------------------------------------  Preprocess via COLMAP  --------------------------------------------
    processed_dir = Path(f'/workspace/data/nerfstudio/{project_name}')
    if processed_dir.exists() and processed_dir.joinpath('transforms.json').exists():
        print('Data already pre-processed. Skipping step!')
    else:
        print(f'Going to process video:{video_path.name}, with command:')
        process_cmd = ['ns-process-data', 'video', '--data', str(video_path), '--output-dir', processed_dir]
        # f'--num-frames-target {video_data.frame_count}')  # add this to process all video frames.
        print(process_cmd)
        try:
            subprocess.check_call(process_cmd)
        except subprocess.CalledProcessError as e:
            print('Error pre-processing data!')
            exit(-1)
    # ---------------------------------------  Train the model on COLMAP output  ---------------------------------
    trained_dir = Path(f'/workspace/outputs/{project_name}/nerfacto')
    train_cmd = ['ns-train', 'nerfacto', '--data', processed_dir, '--project-name', project_name]
    if args.skip_train:
        print('Training model skipped.')
    elif args.train_from_checkpoint:
        print('Training model from checkpoint...')
        path_to_checkpoint = trained_dir.joinpath(get_last_trained_model(trained_dir)).joinpath('nerfstudio_models')
        train_cmd.extend(['--load-dir', path_to_checkpoint])
    else:
        print(f'Going to train NeRF model on COLMAP data, with command:')


    def kill_cond(s: str):
        match = re.search(r'.*100\.\d{1,2}%.*', s)
        if match:
            time.sleep(20)
            return True
        return False


    if not args.skip_train:
        for line in execute_and_track_output(train_cmd, kill_proc_cond=kill_cond):
            print(line, end='')

    # ---------------------------------------  Calculate cut points on the original track  ---------------------
    print('Calculating frame cut points and synthesized poses...')
    last_trained_config_dir = get_last_trained_model(trained_dir)
    fm = FileManager(trained_dir.joinpath(last_trained_config_dir).joinpath('config.yml'))
    cameras_extrinsic = fm.viewer_poses(all_poses=True, update_poses=True)
    poses = Poses(cameras_extrinsic, fps)
    poses.show_poses()
    poses.complete_path()
    poses.show_poses()
    generated_cameras = poses.get_generated_poses()
    cut_interval: tuple[int, int] = poses.get_cut_indices()

    x = 'horizontal' if video_data.height > video_data.width else 'vertical'
    focal_length = fm.load_transforms_file().intrinsics.fl_y if x == 'horizontal' else fm.load_transforms_file().intrinsics.fl_x
    front_edge = max(video_data.height, video_data.width)
    fov = 2 * math.atan(front_edge / (2 * focal_length)) * 180 / math.pi
    cam_path_filename = 'camera_path'
    full_cam_path = fm.generate_cam_path_file(cam_path_filename, *cut_interval, fps=fps, fov=int(fov),
                                              look_at_cameras=generated_cameras)
    print('Camera paths created!')
    # ---------------------------------------  Render synthesized part of the cut-out track  -------------------
    print('Synthesizing missing frames...')
    shutil.rmtree(fm.last_render_dir)
    os.makedirs(fm.last_render_dir)
    render_cmd = ['ns-render', 'camera-path', '--output-format', 'images',
                  '--load-config', str(fm.config_path),
                  '--output-path', str(fm.last_render_dir),
                  '--camera-path-filename', str(full_cam_path)]
    print(f'Synthesizing frames at cut points with cmd:\n{render_cmd}')

    try:
        subprocess.check_call(render_cmd)
    except subprocess.CalledProcessError as e:
        print(f'Command:\n{render_cmd}\n,Failed with return code:{e.returncode}.')
        exit(-1)
    print('Done!')
    # ---------------------------------------  Create a GIF on the final output  -------------------------------
    print('Creating GIF...')
    if args.gif_filename:
        fm.create_gif(args.gif_filename, cut_interval, fps)
    else:
        fm.create_gif(project_name, cut_interval, fps)
    print(f'Done! GIF is created at: {fm.renders_dir.joinpath("gifs")}')
