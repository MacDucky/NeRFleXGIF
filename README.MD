<!-- TOC -->

* [Installation steps](#installation-steps)
    * [Prerequisites:](#prerequisites)
    * [Installation process:](#installation-process)

<!-- TOC -->

# Installation steps

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