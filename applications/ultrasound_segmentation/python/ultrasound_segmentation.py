# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from argparse import ArgumentParser
from threading import Event, Thread

import cupy as cp
import holoscan as hs

from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import (
    AJASourceOp,
    FormatConverterOp,
    HolovizOp,
    InferenceOp,
    SegmentationPostprocessorOp,
    VideoStreamReplayerOp,
)
from holoscan.resources import BlockMemoryPool, CudaStreamPool, MemoryStorageType
from holoscan.gxf import Entity


class OverlayOp(Operator):
    """Operator to overlay information on image"""

    def __init__(self, fragment, *args, **kwargs):
        self.is_running = Event()
        self.out_tensor = None
        self.zero = False
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def stop(self):
        """
        """
        pass

    def overlay(self, cp_tensor):
        """
        """
        # print("Thread started")
        if self.zero:
            cp_tensor.fill(0)
        self.out_tensor = hs.as_tensor(cp_tensor)
        self.is_running.clear()

    def compute(self, op_input, op_output, context):
        # Get input message
        in_message = op_input.receive("in")
        cp_tensor = cp.from_dlpack(in_message.get("out_tensor"))

        if self.zero:
            cp_tensor.fill(0)
        self.out_tensor = hs.as_tensor(cp_tensor)

        # Create output message
        out_message = Entity(context)
        out_message.add(self.out_tensor, "out_tensor")
        op_output.emit(out_message, "out")


class UltrasoundApp(Application):
    def __init__(self, data, source="replayer"):
        """Initialize the ultrasound segmentation application

        Parameters
        ----------
        source : {"replayer", "aja"}
            When set to "replayer" (the default), pre-recorded sample video data is
            used as the application input. Otherwise, the video stream from an AJA
            capture card is used.
        """

        super().__init__()

        # set name
        self.name = "Ultrasound App"

        # Optional parameters affecting the graph created by compose.
        self.source = source

        if data == "none":
            data = os.environ.get("HOLOHUB_DATA_PATH", "../data")

        self.sample_data_path = data

        self.model_path_map = {
            "ultrasound_seg": os.path.join(self.sample_data_path, "us_unet_256x256_nhwc.onnx"),
        }

    def set_overlay_op_parameters(self):
        """Set parameters for the OverlayOperator."""
        if self.overlay_op:
            if self.overlay_op.zero == False:
                self.overlay_op.zero = True
            else:
                self.overlay_op.zero = False

    def compose(self):
        n_channels = 4  # RGBA
        bpp = 4  # bytes per pixel

        cuda_stream_pool = CudaStreamPool(
            self,
            name="cuda_stream",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )

        is_aja = self.source.lower() == "aja"
        if is_aja:
            source = AJASourceOp(self, name="aja", **self.kwargs("aja"))
            drop_alpha_block_size = 1920 * 1080 * n_channels * bpp
            drop_alpha_num_blocks = 2
            drop_alpha_channel = FormatConverterOp(
                self,
                name="drop_alpha_channel",
                pool=BlockMemoryPool(
                    self,
                    storage_type=MemoryStorageType.DEVICE,
                    block_size=drop_alpha_block_size,
                    num_blocks=drop_alpha_num_blocks,
                ),
                cuda_stream_pool=cuda_stream_pool,
                **self.kwargs("drop_alpha_channel"),
            )
        else:
            video_dir = self.sample_data_path
            if not os.path.exists(video_dir):
                raise ValueError(f"Could not find video data: {video_dir=}")
            source = VideoStreamReplayerOp(
                self, name="replayer", directory=video_dir, **self.kwargs("replayer")
            )

        width_preprocessor = 1264
        height_preprocessor = 1080
        preprocessor_block_size = width_preprocessor * height_preprocessor * n_channels * bpp
        preprocessor_num_blocks = 2
        segmentation_preprocessor = FormatConverterOp(
            self,
            name="segmentation_preprocessor",
            pool=BlockMemoryPool(
                self,
                storage_type=MemoryStorageType.DEVICE,
                block_size=preprocessor_block_size,
                num_blocks=preprocessor_num_blocks,
            ),
            cuda_stream_pool=cuda_stream_pool,
            **self.kwargs("segmentation_preprocessor"),
        )

        n_channels_inference = 2
        width_inference = 256
        height_inference = 256
        bpp_inference = 4
        inference_block_size = (
            width_inference * height_inference * n_channels_inference * bpp_inference
        )
        inference_num_blocks = 2
        segmentation_inference = InferenceOp(
            self,
            name="segmentation_inference_holoinfer",
            backend="trt",
            allocator=BlockMemoryPool(
                self,
                storage_type=MemoryStorageType.DEVICE,
                block_size=inference_block_size,
                num_blocks=inference_num_blocks,
            ),
            model_path_map=self.model_path_map,
            pre_processor_map={"ultrasound_seg": ["source_video"]},
            inference_map={"ultrasound_seg": "inference_output_tensor"},
            in_tensor_names=["source_video"],
            out_tensor_names=["inference_output_tensor"],
            enable_fp16=False,
            input_on_cuda=True,
            output_on_cuda=True,
            transmit_on_cuda=True,
        )

        postprocessor_block_size = width_inference * height_inference
        postprocessor_num_blocks = 2
        segmentation_postprocessor = SegmentationPostprocessorOp(
            self,
            name="segmentation_postprocessor",
            allocator=BlockMemoryPool(
                self,
                storage_type=MemoryStorageType.DEVICE,
                block_size=postprocessor_block_size,
                num_blocks=postprocessor_num_blocks,
            ),
            **self.kwargs("segmentation_postprocessor"),
        )

        overlay_op = OverlayOp(
            self,
            name="OverlayOp"
        )
        self.overlay_op = overlay_op

        segmentation_visualizer = HolovizOp(
            self,
            name="segmentation_visualizer",
            cuda_stream_pool=cuda_stream_pool,
            **self.kwargs("segmentation_visualizer"),
        )

        if is_aja:
            self.add_flow(source, segmentation_visualizer, {("video_buffer_output", "receivers")})
            self.add_flow(source, drop_alpha_channel, {("video_buffer_output", "")})
            self.add_flow(drop_alpha_channel, segmentation_preprocessor)
        else:
            self.add_flow(source, segmentation_visualizer, {("", "receivers")})
            self.add_flow(source, segmentation_preprocessor)
        self.add_flow(segmentation_preprocessor, segmentation_inference, {("", "receivers")})
        self.add_flow(segmentation_inference, segmentation_postprocessor, {("transmitter", "")})

        self.add_flow(segmentation_postprocessor, overlay_op, {("", "in")})
        self.add_flow(overlay_op, segmentation_visualizer, {("out", "receivers")})


if __name__ == "__main__":
    config_file = os.path.join(os.path.dirname(__file__), "ultrasound_segmentation.yaml")
    data_file = "/workspace/holohub/data/ultrasound_segmentation"
    app = UltrasoundApp(source="replayer", data=data_file)
    app.config(config_file)
    app.run()
