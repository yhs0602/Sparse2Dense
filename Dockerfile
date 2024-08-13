FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu18.04

# Set environment variable to avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages
RUN apt-get update && apt-get install -y \
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
RUN pip install --no-cache-dir -r requirements.txt
# Install PyTorch and CUDA 11.8 support using conda
RUN conda install -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=11.8 -y

# Install and configure VirtualGL
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
