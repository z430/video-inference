FROM nvcr.io/nvidia/deepstream:7.0-samples-multiarch

RUN apt-get update && apt install --no-install-recommends -y \
    ca-certificates \
    python-gst-1.0 \
    wget

# install vpf / pynvcodec
# https://github.com/NVIDIA/VideoProcessingFramework?tab=readme-ov-file

# install cv cuda
# https://github.com/CVCUDA/CV-CUDA

WORKDIR /app