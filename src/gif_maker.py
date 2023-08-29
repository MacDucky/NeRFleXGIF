# Stolen from :)
# https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python/57751793#57751793

import os
import pathlib
from contextlib import contextmanager, ExitStack

from PIL import Image


@contextmanager
def __push_pop_dir(directory):
    cur_dir = os.getcwd()
    os.chdir(directory)
    yield
    os.chdir(cur_dir)


def create_gif_from_image_dir(image_dir: str | os.PathLike | pathlib.Path, fps: float,
                              output_path: str | os.PathLike | pathlib.Path):
    base_dir = pathlib.Path(image_dir)
    images = os.listdir(image_dir)
    images = [base_dir.joinpath(image) for image in images]
    # use exit stack to automatically close opened images
    with ExitStack() as stack:
        imgs = (stack.enter_context(Image.open(f)) for f in sorted(images))

        # extract  first image from iterator
        img = next(imgs)

        # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
        img.save(fp=output_path, format='GIF', append_images=imgs, save_all=True, duration=1000 / fps, loop=0)


if __name__ == '__main__':
    create_gif_from_image_dir(r'/workspace/data/nerfstudio/poster/images', 24, r'/workspace/renders/poster/test_images/testgifreal.gif')
