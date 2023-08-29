# from extraction_data import extraction_data_transforms
# import tkinter
import time

from src.poses import Poses
from src.file_manager import FileManager
# from numpy import ndarray
import os

# os.system('sudo apt install xvfb')
# os.system('pip3 install PyQt5')
# os.system('pip3 install tk')
# import matplotlib
# # matplotlib.use('Qt5Agg')
# matplotlib.use('TkAgg')

if __name__ == '__main__':
    # process data - both ns-process && get video metadata(fps


    # after train
    fm = FileManager(r'/workspace/outputs/poster/nerfacto/2023-08-18_164728/config.yml')
    # path_transforms = 'data\\transforms.json'
    # camera_intrinsics, cameras_extrinsic = extraction_data_transforms(path_transforms)
    # cameras_extrinsic = np.concatenate((cameras_extrinsic[-2:], cameras_extrinsic[:2]))
    cameras_extrinsic = fm.viewer_poses(all_poses=True, update_poses=True)
    cameras_extrinsic = cameras_extrinsic[:-2]
    # cameras_extrinsic = cameras_extrinsic[::3]
    # todo: calculate from FileManager the fps of the input video and pass it on to poses and save it in filemanager
    poses = Poses(cameras_extrinsic, 24)
    # poses.show_poses()
    poses.complete_path()
    # poses.show_poses()
    generated_cameras = poses.get_generated_poses()
    cut_interval: tuple[int, int] = poses.get_cut_indices()

    fm.generate_cam_path_file('test_corrected', *cut_interval, fov=80, look_at_cameras=generated_cameras)
    # then do:
    # ns-render camera-path --load-config outputs/poster/nerfacto/2023-08-18_164728/config.yml --output-format images --output-path renders/poster/last_render/  --camera-path-filename data/nerfstudio/poster/camera_paths/test_corrected.json
    fm.create_gif()
    i = 0
