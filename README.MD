# NeRFlexGIF

___
In this project, we employed NerfStudio's NeRF framework as a seamless black-box implementation of NeRF technology. Our
aim was to craft flawless circular GIF images from captured videos, achieving a visually captivating result.
___

<!-- TOC -->

* [NeRFlexGIF](#nerflexgif)
    * [Installation steps](#installation-steps)
        * [Prerequisites:](#prerequisites)
            * [Requirements:](#requirements)
        * [Installation process:](#installation-process)
    * [Introduction](#introduction)
    * [Preliminaries](#preliminaries)
        * [NeRF](#nerf)
        * [Camera Extrinsic](#camera-extrinsic)
        * [Cubic Splines](#cubic-splines)
    * [Algorithm](#algorithm)
        * [Cutting The Video](#cutting-the-video)
        * [Generating New Video](#generating-new-video)

<!-- TOC -->

## Installation steps

### Prerequisites:

Any machine with the ability to run docker should suffice.

Throughout this project we've used wsl2 backend+docker desktop for windows. Hence, we will be covering this installation
path.

#### Requirements:

1. Windows OS with wsl2 support.

2. A CUDA capable GPU, see if your GPU is supported [here](https://developer.nvidia.com/cuda-gpus).

   _note: remember the Compute Capability of your GPU, will be needed later on._

___

### Installation process:

1. [Install WSL2 with Ubuntu 22.04](https://linuxconfig.org/ubuntu-22-04-on-wsl-windows-subsystem-for-linux)
2. [Install Docker Desktop](https://docs.docker.com/desktop/install/windows-install/)
3. For debugging:
    1. Edit the provided [env file](.env):
        1. Edit CUDA_ARCHITECTURES to fit for compute capability, e.g. for RTX 30x0 series, CUDA_ARCHITECTURES=86.
    2. Inside [docker-compose.yaml](docker-compose.yml):
        1. Edit mounted volumes under 'volumes:' so that it matches the absolute path mapping from the host machine to
           the
           container (do not edit the container part).
    3. Setup a listener on the docker socket:
        1. <img src="md_resources/docker_connection.png" alt="drawing" width="800"/>
        2. Connection successful should appear when connected, otherwise, try troubleshooting your docker desktop
           software.
    4. Create a remote interpreter based on the docker-compose.yml:
        1. <img src="md_resources/add_interpreter.png" alt="drawing" width="800"/>
        2. Keep pressing next:
           <img src="md_resources/add_interpreter2.png" alt="drawing" width="800"/>
        3. Verify that your python script's run configuration has the docker interpreter:
           <img src="md_resources/verify_interpreter.png" alt="drawing" width="800"/>

4. Run from cmd line:
    ```bash
    docker run --gpus all -v C:\Users\lizas\PycharmProjects\NeRFleXGIF:/workspace/ -v C:\Users\lizas\PycharmProjects\NeRFleXGIF\.cache:/home/user/.cache/ -p 7007:7007 --rm -it --ipc=host dromni/nerfstudio:0.3.2 "python main.py [ARGS]"
    ```
   For help:
    ```bash
    docker run --gpus all -v C:\Users\lizas\PycharmProjects\NeRFleXGIF:/workspace/ -v C:\Users\lizas\PycharmProjects\NeRFleXGIF\.cache:/home/user/.cache/ -p 7007:7007 --rm -it --ipc=host dromni/nerfstudio:0.3.2 "python main.py --help"
    ```

---

## Introduction

Our objective is to create captivating GIFs from static scene videos provided by users. To achieve this, we employ NeRF
technology to seamlessly align the video's starting and ending frames.

The video alignment process involves the removal and generation of frames. We strategically cut frames based on their
camera extrinsic properties within the video. Additionally, we craft a well-defined route through interpolation
techniques and generate new camera extrinsic data accordingly.

Leveraging these camera extrinsic details, we employ our meticulously trained NeRF model to generate new frames. The
result is a flawlessly rendered video that captivates and engages the audience.

## Preliminaries

### NeRF

NeRF, which stands for Neural Radiance Fields, is a cutting-edge computer graphics and computer vision technique
introduced in 2020. It's a neural network-based approach that learns to create highly detailed and realistic 3D scenes
from 2D images. NeRF achieves this by modeling a 3D scene as a continuous function that maps 3D coordinates to color and
opacity values. It takes multiple 2D images of a scene captured from different viewpoints and learns to estimate the
scene's appearance and geometry, allowing for the synthesis of new views of the scene. NeRF has applications in virtual
reality, augmented reality, and 3D scene reconstruction, among others, due to its ability to generate lifelike 3D scenes
from limited image data.

### Camera Extrinsic

The camera extrinsics typically consist of the following components:

1. **Translation Vector (T)**: This vector represents the 3D coordinates of the camera's optical center (also known as
   the camera center) relative to the world coordinate system. It indicates how far the camera is displaced along the x,
   y, and z axes from the origin of the world coordinate system.

2. **Rotation Matrix (R)**: The rotation matrix describes the orientation of the camera in space. It specifies how the
   camera is rotated around each of the coordinate axes (roll, pitch, and yaw). The rotation matrix allows us to
   transform points from the camera's coordinate system to the world coordinate system.

Together, the translation vector (T) and the rotation matrix (R) provide a transformation that maps points from the
world coordinate system into the camera's coordinate system. This transformation is essential for understanding the
camera's viewpoint and perspective when capturing an image or video.

### Cubic Splines

Cubic splines are a mathematical interpolation technique used to create smooth curves or surfaces by connecting a series
of data points with piecewise cubic polynomials. They ensure continuity in both the function's value and its first
derivative at each data point, resulting in a smooth and visually pleasing curve or surface.

## The Algorithm

To make a perfect circular GIF from the video, we need to remove frames from the beginning and end of the video. Then,
we create new frames that follow a path from the end of the video back to the beginning at these cut points.

As NeRF primarily analyzes the scene from a limited set of frames, the synthesized images generated from new camera
positions should closely resemble the originals. If a new position deviates significantly from the original camera's
path, it can lead to the creation of less aesthetically pleasing images.

Let's break down the algorithm into simple steps:

1. Identify the appropriate start and end frames for cutting.
2. Create a smooth path connecting the new endpoint to the new starting point.
3. Determine the number of frames to be synthesized along this path.
4. Synthesize new frames along the path, given the amount of frames.
5. Finally, Merge all frames into a GIF image.

### Cut Point Identification

To cut the video optimally, we want to minimize the following loss function:

```math
cost
```

where $\alpha$ is the angle between start and end point vectors, when projected to the xy plain.

The concept was to preserve the proportion of the average z-height as we moved from the endpoint (end cut
point) to the starting point (first cut point) in the pre-cut video. This approach ensured that the transition between
the start and end maintained a similar "speed" and smoothness to the original frames. The lower the score above, the
smoother the z-height transitions are.
We simply brute-force check a section of frames from the start and end edges as pairs of points for the minimal score.

### Connecting endpoints:

To seamlessly connect the new endpoints, we've used cubic splines to create a polynomial path from the endpoint to the
starting point.

Regarding the positions of the newly positioned camera origins, designated for generating new frames, we employed a
straightforward linear interpolation method for determining the direction vector based on its relative position along
the path.

### Spline sample rate

After determining the remaining number of frames following the cut, we can estimate that the number of generated frames
is proportionate to the number of frames in the complementary $\alpha$ section after the cut. This computation can be
expressed as follows:

```math
number of steps = \lceil{\alpha * \#_frames \over (2\pi - \alpha)}\rceil
```

### Cutting The Video

To cut the frames from the video, we build 2 lines, one from the first and second frames positions, and the other from
the last two frames positions. we calculate the angle between the first frame position and the last line we'll call it
last angle, and
the other side too (between the last position and first line) and we'll call it first angle.
If first angle is smaller than last angle we'll pop from the end, otherwise we will pop the start.
And if from one edge we can't pop we'll try to pop from the other edge.
We pop a frame from an edge if the angle between the position of the frame and the line of the other edge is bigger from
angle between the next position frame and the line other edge.
For robustness we don't pop from an edge if the angle in the first couple of frames grows.

![img_2.png](img_2.png)

```python
    def cut_poses(self):


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
```

```python
    def pop_from_edge(self, is_end: bool) -> bool:


did_pop = False
counter = 1
while not did_pop and counter <= 5:
    angle, next_angle = self.get_angles(is_end, counter - 1)
    if abs(angle) > abs(next_angle) or (angle > 0) ^ (next_angle > 0):
        did_pop = True
        self.pop_cameras(is_end, counter)
    counter += 1
return did_pop
```

the results of the algorithm:

![img.png](img.png)
![img_1.png](img_1.png)

### Generating New Video

To generate new cameras, we need to build a trace dependent on the time of the frame.
we first define the time of the last frame to be 0 and the first to be t, such that t is:

```math
t = {\lceil 2 \cdot fps \cdot {p - p_0 \over v - v_0} \rceil \over fps}
```

After that we put the 2 first frame and 2 last frame (when we know that the tendency between every frame is constant) on
a spline. We sample positions of frames from the spline using the fps of the video.
And the directions of the frames we create from linear interpolation.
