import numpy as np
from numpy import ndarray
from math import pi
from src.look_at_camera import LookAtCamera, creat_direction
from src.spline import Spline
import matplotlib.pyplot as plt


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
        self.look_at_cameras = []
        for camera_extrinsic in cameras_extrinsic:
            self.look_at_cameras.append(LookAtCamera(camera_extrinsic))
        self.og_num_frames = len(self.look_at_cameras)
        self.first_idx = 1
        self.last_idx = len(self.look_at_cameras)
        self.generated_cameras = None

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
            self.look_at_cameras = self.look_at_cameras[:-num_pop_cams]
            self.last_idx -= num_pop_cams
        else:
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

        self.generated_cameras = [None] * sampled_points.shape[0]

        for i, _ in enumerate(self.generated_cameras):
            self.generated_cameras[i] = LookAtCamera.from_position_and_direction(sampled_points[i], sampled_directions[i])

    def complete_path(self):
        self.cut_poses()
        self.generate_poses()
        self.look_at_cameras.extend(self.generated_cameras)
        return self.look_at_cameras

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

    def show_poses(self, boundaries: int = 1):
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
