import math
import os
import re
import shutil
from pathlib import Path

from src.poses import Poses
from src.file_manager import FileManager
from src import gif_maker
from src.video_utils import VideoData
from src import image_utils


def visualization_main(args):
    fm = FileManager('outputs/chair/nerfacto/2023-09-02_163102/config.yml')
    video_path = fm.data_path / 'video' / f'{fm.project_name}.mp4'
    video_data = VideoData(video_path)
    num_frames_target = 300
    spacing = video_data.frame_count // num_frames_target
    number_of_frames = math.ceil(video_data.frame_count / spacing) if spacing > 1 else VideoData.frame_count
    fps = (number_of_frames / video_data.frame_count) * video_data.fps
    poses = Poses(fm.viewer_poses(all_poses=True, update_poses=True), fps, fm.cameras)
    poses.show_poses()
    fullpath_lookats = poses.complete_path()
    poses.show_poses()
    a = poses.get_generated_poses()
    b = poses.get_cut_indices()
    print(b)
    poses.show_all_stages(boundaries=1)

    # merge_two_images('img.png', 'img_1.png', 'output.png')

    if args.interactive:
        print('Generating current position images (interactive mode).')
        poses.generate_intractive_images(fm.interactive_dir, base_image_name='i_image{num}.png')
    else:
        print('Skipped generating positional images. Assuming already exist...')

    assert (fm.renders_dir / "interactive_images").exists(), \
        f'Directory {fm.renders_dir / "interactive_images"} and is required later on'

    if args.inter_gif_name:
        igif_name: str = args.inter_gif_name if args.inter_gif_name.lower().endswith(
            'gif') else args.inter_gif_name + 'gif'
        gif_maker.create_gif_from_image_dir(fm.interactive_dir, fps, fm.interactive_dir / igif_name)
        print(f'Interactive GIF: {igif_name}')

    assert (tmpdir := fm.renders_dir / 'tmpdir').exists() and len(os.listdir(tmpdir)), \
        'Rendered images are not found, please render images via \'create_gif\''

    def list_images(path_to_dir):
        return list(filter(lambda s: s.split('.')[-1] in {'png', 'jpg', 'jpeg'}, os.listdir(path_to_dir)))

    assert len(rendered := list_images(tmpdir)) == len(position_images := list_images(fm.interactive_dir)), \
        'The # of images to be rendered should match # of position images. Are the rendered images present?'

    print('Going to add a border to video+synth images.')
    if (with_border_dir := fm.renders_dir / 'w_border').exists():
        shutil.rmtree(with_border_dir, ignore_errors=True)
    os.mkdir(with_border_dir)
    for image_path, look_at in zip(map(lambda base: tmpdir / base, rendered), fullpath_lookats):
        target_path = with_border_dir / image_path.name
        image_utils.add_border_to_image(image_path, look_at.plot_color, output_path=target_path)

    print('Going to merge interactive images with video+synth images.')
    if (merged_dir := fm.renders_dir / 'merged_images').exists():
        shutil.rmtree(merged_dir, ignore_errors=True)
    os.mkdir(merged_dir)
    index_finder = re.compile(r'\d+')
    index_key_func = lambda f: int(index_finder.search(f).group())
    position_images.sort(key=index_key_func)
    rendered.sort(key=index_key_func)
    for pos_img, border_img in zip(position_images, rendered):
        abs_pos_img = fm.interactive_dir / pos_img
        abs_border_img = with_border_dir / border_img
        image_utils.merge_two_images(abs_pos_img, abs_border_img, merged_dir / abs_border_img.name)
    if args.gif_filepath:
        gif_maker.create_gif_from_image_dir(merged_dir, fps, args.gif_filepath)
    else:
        gif_maker.create_gif_from_image_dir(merged_dir, fps, fm.renders_dir / 'gifs' / 'vis_merged.gif')
