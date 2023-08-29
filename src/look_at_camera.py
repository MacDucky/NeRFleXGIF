import numpy as np
from numpy import ndarray
import matplotlib .pyplot as plt

global_up = np.array([0, 0, 1])


def creat_direction(first_direction, last_direction, relative_time) -> ndarray:
    direction = (1-relative_time) * first_direction + relative_time * last_direction
    return direction / np.linalg.norm(direction)


class LookAtCamera:
    def __init__(self, look_at_cam: ndarray):
        self.look_at_cam = look_at_cam

    @classmethod
    def from_position_and_direction(cls, position: ndarray, backward: ndarray):
        right = np.cross(global_up, backward)
        right = right / np.linalg.norm(right)
        up = np.cross(backward, right)
        up = up / np.linalg.norm(up)

        look_at_cam = np.concatenate((right, up, backward, position))
        look_at_cam = np.reshape(look_at_cam, (4, 3)).T
        look_at_cam = np.concatenate((look_at_cam, np.array([[0, 0, 0, 1]])), axis=0)
        transform = cls(look_at_cam)
        return transform

    def get_position(self):
        return self.look_at_cam[:-1, -1]

    def get_backward(self):
        return self.look_at_cam[:-1, 2]

    def get_up(self):
        return self.look_at_cam[:-1, 1]

    def get_right(self):
        return self.look_at_cam[:-1, 0]

    def plot_look_at_cam(self, ax):
        origin = self.get_position()
        forward = -self.get_backward()
        up = self.get_up()
        right = self.get_right()

        ax.scatter(origin[0], origin[1], origin[2], c='r', marker='o')
        ax.quiver(origin[0], origin[1], origin[2], forward[0], forward[1], forward[2], color='r', length=0.2)
        ax.quiver(origin[0], origin[1], origin[2], up[0], up[1], up[2], color='g', length=0.1)
        ax.quiver(origin[0], origin[1], origin[2], right[0], right[1], right[2], color='b', length=0.1)

    def show_look_at_cam(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        self.plot_look_at_cam(ax)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        plt.show()

