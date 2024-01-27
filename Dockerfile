FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# install ubuntu dependencies
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get update && \
    apt-get -y install python3-pip xvfb ffmpeg git build-essential python-opengl

RUN ln -s /usr/bin/python3 /usr/bin/python
# install python dependencies

# install crafter and griddly environments
RUN git clone https://github.com/FaisalAhmed0/crafter.git
RUN pip install -e crafter/.
RUN pip install griddly 
RUN pip install minatar 
RUN pip install ipython
RUN pip install csv_logger
RUN pip install setuptools 
RUN pip install minigrid 
RUN pip install opencv-python


# install mujoco_py
RUN apt-get -y install wget unzip software-properties-common \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev patchelf

ENV PATH="/root/.local/bin:${PATH}"

COPY ./ /surprise_adaptive_agents
WORKDIR surprise_adaptive_agents
RUN pwd 
RUN pip install -r requirements.txt
RUN pip install -e .
