import numpy as np
from numpy import ndarray
from math import pi
from src.look_at_camera import LookAtCamera, creat_direction, AcceptedColor
from src.spline import Spline
import matplotlib.pyplot as plt
from typing import Literal
from os import PathLike
from PIL import Image
from pathlib import Path

Stage = Literal['pre', 'mid', 'post']


def find_angle(mid_point, point1, point2):
    # Calculate the vectors connecting the points
    vec1 = point1 - mid_point
    vec2 = point2 - mid_point

    # Calculate the dot product of the vectors
    dot_product = np.dot(vec1, vec2)

    # Calculate the magnitudes of the vectors
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)

    # Calculate the cosine of the angle between the vectors
    cos_angle = dot_product / (magnitude1 * magnitude2)

    # Calculate the angle in radians
    angle_radians = np.arccos(cos_angle)
    return angle_radians


class Poses:
    def __init__(self, cameras_extrinsic: ndarray, fps: float):
        self.fps = fps
        self.tendency = 1 / fps
        # after cut is blue
        self.look_at_cameras = []
        # red
        self.popped_out_cams = []
        for camera_extrinsic in cameras_extrinsic:
            self.look_at_cameras.append(LookAtCamera(camera_extrinsic))
        self.og_num_frames = len(self.look_at_cameras)
        self.first_idx = 1
        self.last_idx = len(self.look_at_cameras)
        # green
        self.generated_cameras = None
        self.final_cams = None
        self.cams_to_show = None

    def get_edges_points(self) -> tuple[int, int, int, int]:
        return self.look_at_cameras[0].get_position(), self.look_at_cameras[1].get_position(), self.look_at_cameras[
            -1].get_position(), self.look_at_cameras[-2].get_position()

    def get_edges_angles(self) -> tuple[int, int]:
        first_pnt, second_pnt, last_pnt, last_second_pnt = self.get_edges_points()
        first_angle = pi - find_angle(first_pnt, second_pnt, last_pnt)
        last_angle = pi - find_angle(last_pnt, last_second_pnt, first_pnt)
        return first_angle, last_angle

    def get_points_for_start_angles(self, index: int):
        return self.look_at_cameras[0].get_position(), self.look_at_cameras[1].get_position(), self.look_at_cameras[
            -1 - index].get_position(), self.look_at_cameras[-2 - index].get_position()

    def get_points_for_end_angles(self, index: int):
        return self.look_at_cameras[-1].get_position(), self.look_at_cameras[-2].get_position(), self.look_at_cameras[
            index].get_position(), self.look_at_cameras[index + 1].get_position()

    def get_start_angles(self, index: int) -> tuple[int, int]:
        first_pnt, second_pnt, last_pnt, last_second_pnt = self.get_points_for_start_angles(index)
        angle = pi - find_angle(first_pnt, second_pnt, last_pnt)
        next_angle = pi - find_angle(first_pnt, second_pnt, last_second_pnt)
        return angle, next_angle

    def get_end_angles(self, index: int) -> tuple[int, int]:
        last_pnt, last_second_pnt, first_pnt, second_pnt = self.get_points_for_end_angles(index)
        angle = pi - find_angle(last_pnt, last_second_pnt, first_pnt)
        next_angle = pi - find_angle(last_pnt, last_second_pnt, second_pnt)
        return angle, next_angle

    def get_angles(self, is_start: bool, index: int):
        return self.get_start_angles(index) if is_start else self.get_end_angles(index)

    def pop_cameras(self, is_end: bool, num_pop_cams: int):
        if is_end:
            temp_popped_cams: list[LookAtCamera] = self.look_at_cameras[-num_pop_cams:]
            for cam in temp_popped_cams:
                cam.plot_color = 'r'
            self.popped_out_cams.extend(temp_popped_cams)

            self.look_at_cameras = self.look_at_cameras[:-num_pop_cams]
            self.last_idx -= num_pop_cams
        else:
            temp_popped_cams: list[LookAtCamera] = self.look_at_cameras[:num_pop_cams]
            for cam in temp_popped_cams:
                cam.plot_color = 'r'
            self.popped_out_cams.extend(temp_popped_cams)

            self.look_at_cameras = self.look_at_cameras[num_pop_cams:]
            self.first_idx += num_pop_cams

    def pop_from_edge(self, is_end: bool) -> bool:
        did_pop = False
        counter = 1
        while not did_pop and counter <= np.ceil(self.fps / 5) + 1:
            angle, next_angle = self.get_angles(is_end, counter - 1)
            if abs(angle) > abs(next_angle) or (angle > 0) ^ (next_angle > 0):
                did_pop = True
                self.pop_cameras(is_end, counter)
            counter += 1
        return did_pop

    def cut_poses(self):
        """
        cutting the poses of the cameras which are unnecessary for the newly generated video
        """
        did_pop = True
        while len(self.look_at_cameras) > 4 and did_pop:
            did_pop = False
            first_angle, last_angle = self.get_edges_angles()
            is_pop_from_end = abs(first_angle) < abs(last_angle)

            if is_pop_from_end:
                did_pop = self.pop_from_edge(is_end=True)
            elif not did_pop:
                did_pop = self.pop_from_edge(is_end=False)
                if not is_pop_from_end and not did_pop:
                    did_pop = self.pop_from_edge(is_end=True)

    def time_between_poses(self) -> float:
        """
        finds the time between the last frame and first frame
        :return: the time between the last frame and the first frame
        """

        v = np.linalg.norm(
            self.look_at_cameras[1].get_position() - self.look_at_cameras[0].get_position()) / self.tendency
        v_0 = np.linalg.norm(
            self.look_at_cameras[-1].get_position() - self.look_at_cameras[-2].get_position()) / self.tendency
        x = np.linalg.norm(self.look_at_cameras[0].get_position() - self.look_at_cameras[-1].get_position())
        t = 2 * x / (v + v_0)
        t = np.ceil(t / self.tendency) * self.tendency
        return t

    def generate_poses(self):
        # build the missing path dependent on the time of the video
        first_direction = self.look_at_cameras[-1].get_backward()
        last_direction = self.look_at_cameras[0].get_backward()

        # find the time between
        t = self.time_between_poses()

        # taking two points in the start and two points in the end
        times = np.array([-self.tendency, 0, t, t + self.tendency])
        points = np.array([self.look_at_cameras[-2].get_position(), self.look_at_cameras[-1].get_position(),
                           self.look_at_cameras[0].get_position(), self.look_at_cameras[1].get_position()])
        points = np.array(points)
        spline = Spline(times, points)

        # the time samples
        time_samples = [i * self.tendency for i in range(int(t / self.tendency))][1:]

        # the sampled points on the track
        sampled_points = []
        sampled_directions = []
        for time_sample in time_samples:
            sampled_points.append(spline.sample_point(time_sample))
            sampled_directions.append(creat_direction(first_direction, last_direction, time_sample / t))
        sampled_points = np.array(sampled_points)
        sampled_directions = np.array(sampled_directions)

        self.generated_cameras: list[LookAtCamera] = [None] * sampled_points.shape[0]  # noqa

        for i, _ in enumerate(self.generated_cameras):
            self.generated_cameras[i] = LookAtCamera.from_position_and_direction(sampled_points[i],
                                                                                 sampled_directions[i])
            self.generated_cameras[i].plot_color = 'g'

    def complete_path(self):
        self.cut_poses()
        self.generate_poses()
        self.final_cams = self.look_at_cameras + self.generated_cameras
        self.cams_to_show = self.final_cams
        return self.final_cams

    def get_generated_poses(self) -> list[ndarray]:
        generated_poses = []
        for generated_camera in self.generated_cameras:
            generated_poses.append(generated_camera.look_at_cam)
        return generated_poses

    def get_cut_indices(self) -> tuple[int, int]:
        return self.first_idx, self.last_idx

    def plot_poses(self, ax):
        for look_at_camera in self.look_at_cameras:
            look_at_camera.plot_look_at_cam(ax)

    def plot_positions(self, ax):
        """
        Given a plot, plots the (self) camera origin on the plot.
        :param ax: Plot
        """
        for look_at_camera in self.cams_to_show:
            look_at_camera.plot_origin(ax)

    def generate_interactive_images(self, boundaries: int = 5):
        """
        Yields the post cut stage images origins at current active origin at every active origin (interactive)
        """
        for active_idx, _ in enumerate(self.cams_to_show):
            save_figure_path: str = yield
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlim(-boundaries, boundaries)
            ax.set_ylim(-boundaries, boundaries)
            ax.set_zlim(-boundaries, boundaries)
            for idx, look_at_camera in enumerate(self.cams_to_show):
                look_at_camera.plot_origin(ax, force_color='k' if idx != active_idx else None)

            ax.azim = 45
            ax.elev = 45
            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
            ax.set_zlabel('Z axis')
            plt.savefig(save_figure_path)
            yield f'Saved to {save_figure_path}'

    def show_poses(self, boundaries: int = 5):
        """
        Plot points with camera orientation.
        :param boundaries: xyz axes boundaries in output plot.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-boundaries, boundaries)
        ax.set_ylim(-boundaries, boundaries)
        ax.set_zlim(-boundaries, boundaries)
        self.plot_poses(ax)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        plt.show()

    def plots_points_at_stage(self, stage: Stage, ax):
        match stage:
            case 'pre':
                self.cams_to_show = self.look_at_cameras + self.popped_out_cams
            case 'mid':
                self.cams_to_show = self.look_at_cameras
            case 'post':
                self.cams_to_show = self.final_cams
            case _:
                raise ValueError('stage is not of type Stage')
        self.plot_positions(ax)
        return ax

    def show_positions_at_cut_stage(self, stage: Stage, boundaries: int = 5):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-boundaries, boundaries)
        ax.set_ylim(-boundaries, boundaries)
        ax.set_zlim(-boundaries, boundaries)
        self.plots_points_at_stage(stage, ax)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        plt.show()

    def show_all_stages(self, boundaries: int = 5, save_figure_path: str | None = None):
        stages = ['pre', 'mid', 'post']
        figsize = (25, 10)
        fig, ax = plt.subplots(1, 3, figsize=figsize)
        for i, s in enumerate(stages, start=1):
            scatter_3d_ax = fig.add_subplot(1, 3, i, projection='3d')  # Select the top-left subplot
            scatter_3d_ax.set_xlim(-boundaries, boundaries)
            scatter_3d_ax.set_ylim(-boundaries, boundaries)
            scatter_3d_ax.set_zlim(-boundaries, boundaries)
            scatter_3d_ax.set_title(f'{s.title()} cut stage')
            self.plots_points_at_stage(s, scatter_3d_ax)
            scatter_3d_ax.set_xlabel('X axis')
            scatter_3d_ax.set_ylabel('Y axis')
            scatter_3d_ax.set_zlabel('Z axis')
        plt.tight_layout()
        if save_figure_path is not None:
            plt.savefig(save_figure_path)
        plt.show()


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