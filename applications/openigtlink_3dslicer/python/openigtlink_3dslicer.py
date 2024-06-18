# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import sys

import cupy as cp
import numpy as np
import itk

import holoscan as hs
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.gxf import Entity
from holoscan.resources import UnboundedAllocator

from holohub.openigtlink_tx import OpenIGTLinkTxOp


class LoadMhdOp(Operator):
    """Operator to load mhd file from disk"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out")
        spec.param("file_path", "")

    def compute(self, op_input, op_output, context):
        if not os.path.isfile(self.file_path):
            print("Error: File {} does not exist or is not a file.".format(self.file_path))
            sys.exit()

        # Read image
        # pixel_type = itk.ctype("unsigned char")
        # itk_image = itk.imread(self.file_path, pixel_type)
        itk_image = itk.imread(self.file_path)

        # Get the template of the image
        image_template = itk.template(itk_image)
        print(f"Image Template: {image_template}")

        # Extract the pixel type and dimension
        pixel_type = image_template[1][0]
        dimension = image_template[1][1]
        print(f"Pixel Type: {pixel_type}")
        print(f"Dimension: {dimension}")

        # Get the largest possible region of the image
        region = itk_image.GetLargestPossibleRegion()
        # Get the size of the region
        size = region.GetSize()
        print(f"Size: {size}")

        np_array = itk.GetArrayFromImage(itk_image)
        cp_array = cp.asarray(np_array)

        # Create output message
        out_message = Entity(context)
        # out_message.add(itk_image, "itk_image")
        out_message.add(hs.as_tensor(cp_array), "itk_image")
        op_output.emit(out_message, "out")

class PrintVolumeInfoOp(Operator):
    """Print volume info"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")

    def compute(self, op_input, op_output, context):
        # Get input message
        in_message = op_input.receive("in")

        # Get as numpy array
        np_array = cp.asarray(in_message.get("volume")).get()

        # Print
        print("shape: ", np_array.shape)
        print("mean: ", np.mean(np_array))
        print("min: ", np.min(np_array))
        print("max: ", np.max(np_array))

# class TransformVolumeOp(Operator):
    # """Transform volume"""

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    # def setup(self, spec: OperatorSpec):
    #     spec.input("in")
    #     spec.output("out")

    # def compute(self, op_input, op_output, context):
    #     # Get input message
    #     in_message = op_input.receive("in")

    #     # # Transpose
    #     # tensor = cp.asarray(in_message.get("preprocessed")).get()
    #     # # OBS: Numpy conversion and moveaxis is needed to avoid strange
    #     # # strides issue when doing inference
    #     # tensor = np.moveaxis(tensor, 2, 0)[None]
    #     # tensor = cp.asarray(tensor)

    #     # Create output message
    #     out_message = Entity(context)
    #     out_message.add(hs.as_tensor(tensor), "volume")
    #     op_output.emit(out_message, "out")


class OpenIGTLinkApp(Application):
    def __init__(self):
        """Initialize the endoscopy tool tracking application"""
        super().__init__()

        self.name = "Endoscopy App"

        self.data_path = "/workspace/holohub/data/colonoscopy_segmentation"

    def compose(self):
        pool = UnboundedAllocator(self, name="pool")

        load_mhd = LoadMhdOp(
            self,
            name="load_mhd",
            pool=pool,
            **self.kwargs("load_mhd"),
        )

        # transform_volume = TransformVolumeOp(
        #     self,
        #     name="transform_volume",
        #     pool=pool,
        # )

        print_volume_info = PrintVolumeInfoOp(
            self,
            name="print_volume_info",
            pool=pool,
        )

        openigtlink_tx = OpenIGTLinkTxOp(
            self,
            name="openigtlink_tx",
            **self.kwargs("openigtlink_tx")
        )

        # Build flow
        # self.add_flow(load_mhd, print_volume_info, {("out", "in")})

        self.add_flow(load_mhd, openigtlink_tx, {("out", "receivers")})

        # self.add_flow(load_mhd, transform_volume, {("out", "in")})
        # self.add_flow(transform_volume, openigtlink_tx, {("out", "receivers")})


if __name__ == "__main__":
    config_file = os.path.join(os.path.dirname(__file__), "openigtlink_3dslicer.yaml")

    app = OpenIGTLinkApp()
    app.config(config_file)
    app.run()
