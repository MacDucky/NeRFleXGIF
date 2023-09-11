# NeRFlexGIF

<!-- TOC -->

* [NeRFlexGIF](#nerflexgif)
    * [Installation steps](#installation-steps)
        * [Prerequisites:](#prerequisites)
        * [Installation process:](#installation-process)
    * [Introduction](#introduction)
    * [Preliminaries](#preliminaries)
        * [Camera Extrinsic](#camera-extrinsic)
        * [Spline](#spline)
    * [Algorithm](#algorithm)
        * [Cutting The Video](#cutting-the-video)
        * [Generating New Video](#generating-new-video)

<!-- TOC -->

## Installation steps

### Prerequisites:

Any machine with the ability to run docker should suffice.

Throughout this project we've used wsl2+docker from apt package manager, hence we will be covering this installation
path.

1. An Ubuntu machine (recommended distribution is 22.04) OR wsl2 with an Ubuntu 22.04 distribution

   _note: wsl2 with Ubuntu 22.04 is important since it provides CUDA and systemd support_

2. A CUDA capable GPU, see if your GPU is supported [here](https://developer.nvidia.com/cuda-gpus).

   _note: remember the Compute Capability of your GPU, might be needed later on._

___

### Installation process:

1. [Install docker engine](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository)

2. [Docker post installation steps](https://docs.docker.com/engine/install/linux-postinstall/)

   _note: if you are using wsl, logging out means shutting down and restarting wsl,_
   _from cli: ``` wsl --shutdown ``` and then ``` wsl.exe ```_

---

## Introduction

Given a video by the user of a static scene, Our goal is to generate new video such that we align the start and the end
of the video frames using NeRF to generate flawless GIFs.
To align the video we cut frames from the video and generate new ones.
We cut frames according to their cameras extrinsic from the video and generating new cameras extrinsic according to the
route we creat using interpolations.
And using the cameras extrinsic we generate new frames using our trained NeRF to complete the video, such it will be
flawless.

## Preliminaries

### NeRF

NeRF is a

### Camera Extrinsic

Camera extrinsic is a 4 by 4 matrix representing the position and direction of the camera in our space.

### Spline

spline interpolation is a form of interpolation where the interpolant is a special type of piecewise polynomial called a
spline.

## Algorithm

For all the algorithm we look at the frames as their cameras extrinsic.

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
t = \lceil 2 \cdot (p - p_0) / (v - v_0) * fps \rceil / fps
```

After that we put the 2 first frame and 2 last frame (when we know that the tendency between every frame is constant) on
a spline. We sample positions of frames from the spline using the fps of the video.
And the directions of the frames we create from linear interpolation.
