import math
import subprocess

from pathlib import Path
from argparse import ArgumentParser

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


def parsed_args():
    parser = ArgumentParser('NerFlexGif', description='A perfect GIF creator.',
                            epilog="Flow: 1. Process video (COLMAP). 2. Train NeRF model."
                                   "3. Crop excess parts. 4. Synthesize middle frames. 5. Generate GIF!")
    parser.add_argument('-v', '--video-path', type=path, help="Path to video file to process.",
                        required=True)
    parser.add_argument('-p', '--project-name', type=str, help="Name of project.")
    return parser.parse_args()


if __name__ == '__main__':
    num_frames_target = 300

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

    processed_dir = Path(f'/workspace/data/nerfstudio/{project_name}')
    if processed_dir.exists() and processed_dir.joinpath('transforms.json').exists():
        print('Data already pre-processed. Skipping step!')
    else:
        print(f'Going to process video:{video_path.name}, with command:\n')
        process_cmd = (f'ns-process-data video --data {video_path} --output-dir {processed_dir}')
        # f'--num-frames-target {video_data.frame_count}')  # add this to process all video frames.
        print(process_cmd.split())
        try:
            subprocess.check_call(process_cmd.split())
        except subprocess.CalledProcessError as e:
            print('Error pre-processing data!')
            exit(-1)

    # trained_basecmd = ['ns-train', 'nerfacto']
    # if trained_dir.exists() and (last_trained_config_dir := get_last_trained_model(trained_dir)) != '':
    #     train_subcmd = ['--load-config', f'{last_trained_config_dir.joinpath("config.yml")}']
    # else:
    #     train_subcmd = ['--data', processed_dir, '--project-name', project_name]
    #
    # train_fullcmd = trained_basecmd + train_subcmd
    # try:
    #     subprocess.check_call(train_fullcmd)
    # except subprocess.CalledProcessError as e:
    #     print('Error pre-processing data!')
    #     exit(-1)

    # todo render here!
    # example:
    # 'ns-render camera-path --load-config outputs/poster/nerfacto/2023-08-18_164728/config.yml --output-format images --output-path renders/poster/last_render/  --camera-path-filename data/nerfstudio/poster/camera_paths/test_corrected.json'

    # train_cmd = 'ns-train nerfacto --data data/nerfstudio/chair --project-name chair'
    # after train
    trained_dir = Path(f'/workspace/outputs/{project_name}/nerfacto')
    last_trained_config_dir = get_last_trained_model(trained_dir)
    fm = FileManager(trained_dir.joinpath(last_trained_config_dir).joinpath('config.yml'))
    # path_transforms = 'data\\transforms.json'
    # camera_intrinsics, cameras_extrinsic = extraction_data_transforms(path_transforms)
    # cameras_extrinsic = np.concatenate((cameras_extrinsic[-2:], cameras_extrinsic[:2]))
    cameras_extrinsic = fm.viewer_poses(all_poses=True, update_poses=True)
    cameras_extrinsic = cameras_extrinsic[:-2]
    # cameras_extrinsic = cameras_extrinsic[::3]

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
    fm.generate_cam_path_file('test_corrected', *cut_interval, fps=fps, fov=int(fov), look_at_cameras=generated_cameras)
    # then do:
    # ns-render camera-path --load-config outputs/poster/nerfacto/2023-08-18_164728/config.yml --output-format images --output-path renders/poster/last_render/  --camera-path-filename data/nerfstudio/poster/camera_paths/test_corrected.json
    fm.create_gif(f'{project_name}', cut_interval, fps)
    i = 0
