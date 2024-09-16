import glob
import itertools
import logging
import os
import sys
from fractions import Fraction
from pathlib import Path
from typing import List, Union

import av
import cvcuda
import numpy as np
import nvcv
import PyNvCodec as nvc
import PyNvVideoCodec as nvvc
import PytorchNvCodec as pnvc
import torch
from loguru import logger
from nvidia import nvimgcodec

from .utils import nvtx_range

pixel_format_to_cvcuda_code = {
    nvvc.Pixel_Format.YUV444: cvcuda.ColorConversion.YUV2RGB,
    nvvc.Pixel_Format.NV12: cvcuda.ColorConversion.YUV2RGB_NV12,
}


class Batch:
    def __init__(
        self,
        batch_idx: int,
        data: Union[cvcuda.Tensor, np.ndarray, torch.Tensor],
        fileinfo: Union[str, List[str]],
    ) -> None:
        self.batch_idx = batch_idx
        self.data = data
        self.fileinfo = fileinfo


class VideoBatchDecoder:
    def __init__(
        self,
        input_path,
        batch_size,
        device_id,
        cuda_ctx,
        cuda_stream,
        cvcuda_perf,
    ) -> None:
        self.input_path = input_path
        self.batch_size = batch_size
        self.device_id = device_id
        self.cuda_ctx = cuda_ctx
        self.cuda_stream = cuda_stream
        self.cvcuda_perf = cvcuda_perf

        self.total_decoded = 0
        self.batch_idx = 0
        self.decoder = None
        self.cvcuda_RGBtensor_batch = None
        nvDemux = nvcc.PyNvDemux(self.input_path)
        self.fps = nvDemux.Framerate()

        logger.info(f"Using PyNvVideoCodec decoder version: {nvvc.__version__}")

    def __cal__(self):
        with nvtx_range("decoder.PyVideoCodec"):
            if self.decoder is None:
                self.decoder = nvc.PyNvDecoder(
                    self.input_path,
                    self.device_id,
                    self.cuda_ctx,
                    self.cuda_stream,
                )

            cvcuda_YUVtensor = self.decoder.get_next_frame(self.batch_size)
            if cvcuda_YUVtensor is None:
                return None

            # cvcuda_code = pix
