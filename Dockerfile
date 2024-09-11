FROM nvcr.io/nvidia/deepstream:7.0-triton-multiarch

RUN mkdir /app
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    build-essential \
    ca-certificates libsm6 libxext6 curl ffmpeg 'libsm6' 'libxext6' git \
    build-essential cmake pkg-config unzip yasm git checkinstall \
    libjpeg-dev libpng-dev libtiff-dev libunistring-dev libx265-dev libnuma-dev\
    libavcodec-dev libavformat-dev libswscale-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libxvidcore-dev x264 libx264-dev libfaac-dev libmp3lame-dev libtheora-dev \
    libfaac-dev libmp3lame-dev libvorbis-dev libgtk-3-dev libatlas-base-dev gfortran \
    libtool libc6 libc6-dev wget libnuma-dev libgtk2.0-dev libgstrtspserver-1.0-dev gstreamer1.0-rtsp sudo tmux \
    && rm -rf /var/lib/apt/lists/*

# install vpf / pynvcodec
# https://github.com/NVIDIA/VideoProcessingFramework?tab=readme-ov-file

# install cv cuda
# https://github.com/CVCUDA/CV-CUDA
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

WORKDIR /app
