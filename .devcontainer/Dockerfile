FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04
# FROM stablebaselines/stable-baselines

RUN apt update
RUN apt install -y libopenmpi-dev git
# python3.11 python3.11-venv
# RUN apt install -y  libcufft10 libcublas11 libcudnn8

ARG USERNAME=nonroot
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN apt install -y task-spooler htop nvtop btop python3.11 python3.11-venv libgl1 libglib2.0-0

USER $USERNAME
