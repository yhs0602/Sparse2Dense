FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu18.04

# Set environment variable to avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

ADD busybox-static_1.30.1-4_amd64.deb /tmp

# install busybox from deb
RUN dpkg -i /tmp/busybox-static_1.30.1-4_amd64.deb

# https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212772
RUN apt-key del 7fa2af80
RUN rm -rf /var/lib/apt/lists/*
# Remove 'Signed-By' and NVIDIA repository entries from the sources lists
RUN sed -i '/Signed-By/d' /etc/apt/sources.list.d/cuda.list \
    && sed -i '/developer\.download\.nvidia\.com\/compute\/cuda\/repos/d' /etc/apt/sources.list
RUN sed -i '/developer\.download\.nvidia\.com\/compute\/cuda\/repos/d' /etc/apt/sources.list.d/*
RUN sed -i '/developer\.download\.nvidia\.com\/compute\/machine-learning\/repos/d' /etc/apt/sources.list.d/*

RUN rm -rf /var/lib/apt/lists/*
RUN rm -rf /etc/apt/sources.list.d/*
RUN rm -rf /etc/apt/sources.list

# Add Ubuntu 18.04 (Bionic Beaver) default repositories
RUN echo "deb http://archive.ubuntu.com/ubuntu/ bionic main restricted universe multiverse" > /etc/apt/sources.list \
    && echo "deb http://archive.ubuntu.com/ubuntu/ bionic-updates main restricted universe multiverse" >> /etc/apt/sources.list \
    && echo "deb http://archive.ubuntu.com/ubuntu/ bionic-backports main restricted universe multiverse" >> /etc/apt/sources.list \
    && echo "deb http://archive.ubuntu.com/ubuntu/ bionic-security main restricted universe multiverse" >> /etc/apt/sources.list

RUN /bin/busybox wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb


# https://askubuntu.com/a/1228775/901082
# Install necessary packages
RUN apt-get update -o Acquire::CompressionTypes::Order::=gz --fix-missing && apt-get install -y \
    libglew-dev \
    cmake \
    libpng-dev \
    xvfb \
    wget \
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Add conda to the PATH environment variable
ENV PATH=/opt/conda/bin:$PATH

# Create a conda environment with Python 3.9
RUN conda create -n minecraft_maze python=3.9 -y
# Set the shell to use conda environment
SHELL ["conda", "run", "-n", "minecraft_maze", "/bin/bash", "-c"]

# Install OpenJDK 21 using conda
RUN conda install -c conda-forge openjdk=21 -y
# Update conda
RUN conda update -n base -c defaults conda -y
# Copy requirements file and install Python packages
COPY requirements.txt .
RUN pip install -r requirements.txt # --no-cache-dir
# Install PyTorch and CUDA 11.8 support using conda
RUN conda install -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=11.8 -y

# Install and configure VirtualGL
RUN apt update && apt install xserver-xorg-core x11-xserver-utils libxtst6 libxv1 libegl1 -y

RUN wget -O vgl3.1.deb https://sourceforge.net/projects/virtualgl/files/3.1/virtualgl_3.1_amd64.deb/download
RUN dpkg -i vgl3.1.deb
RUN vglserver_config -config +s +f -y

# Set up Xvfb and VirtualGL configuration
RUN Xvfb :2 -screen 0 1024x768x24 +extension GLX -ac +extension RENDER &
ENV DISPLAY=:2

# Configure X11 screen settings and environment variables
RUN xset s off
RUN xset -dpms
RUN xset q

# Set the PYTHONPATH environment variable
ENV PYTHONPATH=.

# Set the entry point to run the Python script
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "minecraft_maze", "python", "room/experiments/sparse.py", "--port1", "31001", "--device-id", "0"]
