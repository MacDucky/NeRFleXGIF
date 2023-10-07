from typing import Literal
from pathlib import Path
from os import PathLike
from PIL import Image


def merge_two_images(image_path1: str | PathLike, image_path2: str | PathLike, output_path: str | PathLike):
    # Load the two images

    with Image.open(Path(image_path1)) as image1, Image.open(Path(image_path2)) as image2:
        # Ensure they have the same height
        height = max(image1.height, image2.height)

        # Define the separation width
        separation_width = 20  # Adjust this value as needed

        # Calculate the width of the new image
        new_width = image1.width + separation_width + image2.width

        # Create a new blank image with the combined width and height
        new_image = Image.new('RGB', (new_width, height))

        # Paste the first image on the new image
        new_image.paste(image1, (0, 0))

        # Paste the second image with separation space
        new_image.paste(image2, (image1.width + separation_width, 0))

        # Save the resulting image
        new_image.save(Path(output_path))


def add_border_to_image(image_path, border_color=Literal['b', 'g'], border_width=10, output_path=None):
    """
    Adds a border of the specified color and width to an image.

    Args:
        image_path (str): Path to the input image.
        border_color: RGB color tuple for the border (default is red).
        border_width (int): Width of the border in pixels (default is 10).
        output_path (str): Optional path to save the resulting image. If None, the image won't be saved.

    Returns:
        PIL.Image.Image: Image object with the added border.
    """
    # Open the image using Pillow
    img = Image.open(image_path)

    # Calculate the new image size with the border
    new_width = img.width + 2 * border_width
    new_height = img.height + 2 * border_width

    color_dict = {'b': (0, 0, 178), 'g': (0, 128, 0)}
    # Create a new image with the larger size and fill it with the border color
    bordered_img = Image.new('RGB', (new_width, new_height), color_dict.get(str(border_color), (255, 0, 0)))

    # Paste the original image onto the center of the new image
    bordered_img.paste(img, (border_width, border_width))

    # Save the resulting image if an output path is provided
    if output_path:
        bordered_img.save(output_path)

    return bordered_img
