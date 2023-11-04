# Stolen from :)
# https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python/57751793#57751793

import os
import re
import pathlib
from contextlib import ExitStack
from PIL import Image


def create_gif_from_image_dir(image_dir: str | os.PathLike | pathlib.Path, fps: float,
                              output_path: str | os.PathLike | pathlib.Path):
    base_dir = pathlib.Path(image_dir)
    images = list(filter(lambda f: not f.endswith('gif'), os.listdir(image_dir)))
    index_finder = re.compile(r'\d+')
    images.sort(key=lambda f: int(index_finder.search(str(f)).group()))
    images = [base_dir.joinpath(image) for image in images]
    # use exit stack to automatically close opened images
    with ExitStack() as stack:
        imgs = (stack.enter_context(Image.open(f)) for f in images)

        # extract  first image from iterator
        img = next(imgs)

        # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
        img.save(fp=output_path, format='GIF', append_images=imgs, save_all=True, duration=1000 / fps, loop=0)
