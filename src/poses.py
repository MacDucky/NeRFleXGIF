import os

import torch
import numpy as np
from numpy import ndarray
from math import pi
from src.look_at_camera import LookAtCamera, create_direction, AcceptedColor
from src.spline import Spline
import matplotlib.pyplot as plt
from typing import Literal
from os import PathLike
from PIL import Image
from pathlib import Path
from itertools import product
import shutil
from nerfstudio.cameras.camera_utils import get_interpolated_poses, get_interpolated_poses_many

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
    def __init__(self, cameras_extrinsic: ndarray, fps: float, cameras=None):
        self.fps = fps
        self.tendency = 1 / fps
        # after cut is blue
        self.look_at_cameras = []
        # red
        self.popped_out_cams = []
        self.cameras = cameras
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
        self.__cut_vertical_poses()

        # self.__cut_vertical_poses()
        # pop_from_edges = int(len(self.look_at_cameras) * 0.07)
        # # self.pop_cameras(True, pop_from_edges)
        # self.pop_cameras(False, int(len(self.look_at_cameras) * 0.3))

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

    def generate_poses(self, default_method: bool = False):
        first_direction = self.look_at_cameras[-1].get_backward()
        last_direction = self.look_at_cameras[0].get_backward()

        if default_method:
            # build the missing path dependent on the time of the video
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
                sampled_directions.append(create_direction(first_direction, last_direction, time_sample / t))
            sampled_points = np.array(sampled_points)
            sampled_directions = np.array(sampled_directions)

            self.generated_cameras: list[LookAtCamera] = [None] * sampled_points.shape[0]  # noqa

            for i, _ in enumerate(self.generated_cameras):
                self.generated_cameras[i] = LookAtCamera.from_position_and_direction(sampled_points[i],
                                                                                     sampled_directions[i])
                self.generated_cameras[i].plot_color = 'g'
        else:
            radial_alpha = self.__calculate_radial_angle(self.look_at_cameras[-1].get_position()[:2],
                                                         self.look_at_cameras[0].get_position()[:2])
            # sample num is proportional to "existing" deg num of points
            num_of_steps = int(np.ceil(radial_alpha * len(self.look_at_cameras) / (2 * np.pi - radial_alpha)))

            points = np.array([self.look_at_cameras[-2].get_position(), self.look_at_cameras[-1].get_position(),
                               self.look_at_cameras[0].get_position(), self.look_at_cameras[1].get_position()])
            times = np.array([-self.tendency, 0, self.tendency * num_of_steps, (self.tendency + 1) * num_of_steps])

            spline = Spline(times, points)

            samples_percentile = 0.3 if num_of_steps > 30 else 0.5
            time_samples = np.linspace(0, self.tendency * num_of_steps, int(np.ceil(num_of_steps * samples_percentile)))
            position_samples = [spline.sample_point(t_point) for t_point in time_samples[1:-1]]
            time_samples = time_samples / max(time_samples)
            backward_samples = [create_direction(first_direction, last_direction, t) for t in time_samples[1:-1]]
            up_samples = [create_direction(self.look_at_cameras[-1].get_up(), self.look_at_cameras[0].get_up(), t) for t
                          in time_samples[1:-1]]
            generated_key_poses = [LookAtCamera.from_position_and_direction(pos, direction, up).look_at_cam[:-1, :] for
                                   pos, direction, up in zip(position_samples, backward_samples, up_samples)]
            generated_key_poses = ([self.look_at_cameras[-2].look_at_cam[:-1, :],
                                    self.look_at_cameras[-1].look_at_cam[:-1, :]] +
                                   generated_key_poses +
                                   [self.look_at_cameras[0].look_at_cam[:-1, :],
                                    self.look_at_cameras[1].look_at_cam[:-1, :]])
            # generated_key_poses = [self.look_at_cameras[-2].look_at_cam[:-1, :]] + generated_key_poses + [
            #     self.look_at_cameras[2].look_at_cam[:-1, :]]
            steps_per_transition = int(np.ceil(int(num_of_steps / len(generated_key_poses))))
            cam_intrinsics = self.cameras.get_intrinsics_matrices()

            generated_cameras, _ = get_interpolated_poses_many(
                poses=torch.from_numpy(np.array(generated_key_poses)),  # maybe need to remove 1 row in the end
                Ks=cam_intrinsics[:len(generated_key_poses), :],
                steps_per_transition=steps_per_transition,
                order_poses=True
            )

            # remove duplicated frames,
            # resulting from generated_key_poses (done deliberately to smooth transition at starting/endpoints
            generated_cameras = generated_cameras[steps_per_transition:-steps_per_transition]
            # generated_cameras = get_interpolated_poses(self.look_at_cameras[-1].look_at_cam,
            #                                            self.look_at_cameras[0].look_at_cam, steps=round(num_of_steps))
            full_cams = []
            for cam in generated_cameras:
                t_to_np = cam.numpy()
                homogeneous_cam = np.vstack([t_to_np, np.atleast_2d([0, 0, 0, 1])])
                full_cams.append(LookAtCamera(homogeneous_cam, 'g'))
            self.generated_cameras = full_cams

    def complete_path(self):
        self.cut_poses()
        # self.generate_poses(default_method=True)
        self.generate_poses()
        self.final_cams = self.look_at_cameras + self.generated_cameras
        self.cams_to_show = self.final_cams
        return self.final_cams

    def get_generated_poses(self) -> list[ndarray]:
        return [generated_cam.look_at_cam for generated_cam in self.generated_cameras]

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

    def __generate_interactive_images(self, boundaries: int = 5):
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
            plt.close()
            yield f'Saved to {save_figure_path}'

    def generate_intractive_images(self, output_dir: str | Path | os.PathLike,
                                   base_image_name: str = 'image_{num}.png'):
        output_dir = Path(output_dir)
        shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(exist_ok=True)
        base_image_path = str(output_dir / base_image_name)
        num = 0
        generator = self.__generate_interactive_images(boundaries=1)
        max_digits = len(str(len(self.cams_to_show)))
        try:
            while True:
                next(generator)
                s_num = str(num).zfill(max_digits)
                answer = generator.send(base_image_path.format(num=s_num))
                num += 1
                print(answer)
        except StopIteration:
            print('Done generating images.')

    def show_poses(self, boundaries: int = 1):
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

    def show_positions_at_cut_stage(self, stage: Stage, boundaries: int = 1):
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
        plt.close()

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
            scatter_3d_ax.view_init(elev=20, azim=70)
        plt.tight_layout()
        if save_figure_path is not None:
            plt.savefig(save_figure_path)

        plt.show()
        plt.close()

    def __calculate_vertical_deltas(self, cameras) -> np.ndarray:
        # abs(dz/da)
        deltas = []
        for pair in zip(cameras, cameras[1:]):
            dz = abs(pair[1].get_position()[2] - pair[0].get_position()[2])
            da = self.__calculate_radial_angle(pair[1].get_position(), pair[0].get_position())
            deltas.append(dz / da)
        return np.array(deltas)
        # return np.array(deltas) ** 2

    @staticmethod
    def __calculate_radial_angle(vec1: np.ndarray, vec2: np.ndarray):
        return np.arccos(np.dot(vec1[:2] / np.linalg.norm(vec1[:2]), vec2[:2] / np.linalg.norm(vec2[:2])))

    def __calculate_best_vertical_candidates(self):
        num_cameras = len(self.look_at_cameras)
        edges_percentile = int(num_cameras * 0.2)
        first_percentiles = range(edges_percentile)
        last_percentiles = range(num_cameras - edges_percentile, num_cameras)
        last_percentiles = reversed(list(last_percentiles))
        all_deltas = self.__calculate_vertical_deltas(self.look_at_cameras)

        best_s_idx = 0
        best_f_idx = num_cameras - 1
        best_curr_pair = best_s_idx, best_f_idx

        best_score = np.inf

        all_degs = self.__calculate_all_radial_degrees()
        for s_idx, f_idx in product(first_percentiles, last_percentiles):
            sum_deg = np.sum(all_degs[s_idx:f_idx])
            if sum_deg < np.pi * 2:
                # Compute current pair score
                curr_frames_left: int = f_idx - s_idx

                # calc angles
                radial_angle = self.__calculate_radial_angle(self.look_at_cameras[s_idx].get_position(),
                                                             self.look_at_cameras[f_idx].get_position())
                c_angle = np.pi * 2 - radial_angle

                frames_to_complete: int = (curr_frames_left / c_angle) * radial_angle

                start_end_delta = (abs(self.look_at_cameras[f_idx].get_position()[2] -
                                       self.look_at_cameras[s_idx].get_position()[2])) / radial_angle
                sum_current_deltas = np.sum(all_deltas[s_idx:f_idx])

                # beta is weighted score
                # beta = ((num_cameras + 1 - f_idx + s_idx) / num_cameras)
                # current_score = ((sum_current_deltas / c_angle) - (start_end_delta_squared / radial_angle))
                # current_score = abs(c_angle ** 2 * sum_current_deltas - radial_angle ** 2 * start_end_delta_squared)
                current_score = abs(sum_current_deltas / curr_frames_left - start_end_delta)
                # current_score = abs(sum_current_deltas*frames_to_complete - start_end_delta*curr_frames_left)

                # check if score sufficent, if not check if gives better result then current best score
                if current_score < best_score:
                    best_score = current_score
                    best_curr_pair = s_idx, f_idx
            # if current_score > 0:
            #     break
        return best_curr_pair

    def __cut_vertical_poses(self):
        best_vertical_cut_pair = self.__calculate_best_vertical_candidates()
        pop_camera_from_start = best_vertical_cut_pair[0]
        pop_camera_from_end = self.last_idx - self.first_idx - best_vertical_cut_pair[1]
        self.pop_cameras(False, pop_camera_from_start)
        self.pop_cameras(True, pop_camera_from_end)

    def __calculate_all_radial_degrees(self):
        all_cams = self.look_at_cameras
        all_degs = []
        for curr, nxt in zip(all_cams, all_cams[1:]):
            all_degs.append(self.__calculate_radial_angle(curr.get_position(), nxt.get_position()))
        return np.array(all_degs)
