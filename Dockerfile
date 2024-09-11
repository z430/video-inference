FROM nvcr.io/nvidia/deepstream:7.0-triton-multiarch

RUN mkdir /app
WORKDIR /app

RUN apt-get update && apt install --no-install-recommends -y \
    ca-certificates \
    wget

# use conda to simplify some dependency managemeny
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

ENV PATH /opt/conda/bin:${PATH}

# install vpf / pynvcodec
# https://github.com/NVIDIA/VideoProcessingFramework?tab=readme-ov-file

# install cv cuda
# https://github.com/CVCUDA/CV-CUDA
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

WORKDIR /app
