# Base container that includes all dependencies but not the actual repo

ARG UBUNTU_VERSION=18.04
ARG ARCH=
ARG CUDA=10.0

#FROM nvidia/cudagl${ARCH:+-$ARCH}:${CUDA}-base-ubuntu${UBUNTU_VERSION} as base
#FROM nvidia/vulkan:1.1.121 as base
FROM nvidia/cudagl:11.4.2-base-ubuntu20.04 as base
# ARCH and CUDA are specified again because the FROM directive resets ARGs
# (but their default value is retained if set previously)

ARG UBUNTU_VERSION
ARG ARCH
ARG CUDA
ARG CUDNN=7.6.5.32-1

SHELL ["/bin/bash", "-c"]


ENV DEBIAN_FRONTEND="noninteractive"
# See http://bugs.python.org/issue19846
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# install anaconda
RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 libc6 libvulkan1 vulkan-utils\
    git mercurial subversion
    
# NOTE: we don't use TF so might not need some of these
# ========== Tensorflow dependencies ==========
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        pkg-config \
        software-properties-common \
        zip \
        unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]

RUN apt-get update -y
# RUN apt-get install -y python3-dev python3-pip
RUN apt-get update --fix-missing
RUN apt-get install -y wget bzip2 ca-certificates git vim
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        premake4 \
        git \
        curl \
        vim \
        ffmpeg \
	    libgl1-mesa-dev \
	    libgl1-mesa-glx \
	    libglew-dev \
	    libosmesa6-dev \
	    libxrender-dev \
	    libsm6 libxext6 \
        unzip \
        patchelf \
        ffmpeg \
        libxrandr2 \
        libxinerama1 \
        libxcursor1 \
        python3-dev python3-pip graphviz \
        freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev libglew1.6-dev mesa-utils
        
# Not sure why this is needed
ENV LANG C.UTF-8

# Not sure what this is fixing
# COPY ./files/Xdummy /usr/local/bin/Xdummy
# RUN chmod +x /usr/local/bin/Xdummy
        
ENV PATH /opt/conda/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda2-2019.10-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /etc/bash.bashrc

RUN conda update -y --name base conda && conda clean --all -y

RUN conda create --name surprise-adapt python=3.7 pip

RUN echo "source activate surprise-adapt" >> ~/.bashrc
ENV PATH /opt/conda/envs/surprise-adapt/bin:$PATH

RUN mkdir /root/playground

# make sure your domain is accepted
# RUN touch /root/.ssh/known_hosts
RUN mkdir /root/.ssh
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts

RUN ls
WORKDIR /root/playground/
RUN git clone https://github.com/Neo-X/rlkit.git
WORKDIR /root/playground/rlkit
RUN git checkout surprise
RUN git reset --hard origin/surprise
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        swig
RUN pip install -r requirements.txt


## Install VizDoom dependancies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
build-essential zlib1g-dev libsdl2-dev libjpeg-dev \
nasm tar libbz2-dev libgtk2.0-dev cmake git libfluidsynth-dev libgme-dev \
libopenal-dev timidity libwildmidi-dev unzip libboost-all-dev liblua5.1-dev

RUN ls

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN conda install -n surprise-adapt pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
RUN conda install -n surprise-adapt x264=='1!152.20180717' ffmpeg=4.0.2 -c conda-forge

ENV IMAGEIO_FFMPEG_EXE="/usr/bin/ffmpeg"

RUN pip install griddly==1.6.0

WORKDIR /root/playground

ENV IMAGEIO_FFMPEG_EXE="/usr/bin/ffmpeg"
RUN ls
