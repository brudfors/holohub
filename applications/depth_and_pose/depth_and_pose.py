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
from argparse import ArgumentParser

import cupy as cp
import cv2
import holoscan as hs
import numpy as np
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.gxf import Entity
from holoscan.operators import (
    FormatConverterOp,
    HolovizOp,
    InferenceOp,
    V4L2VideoCaptureOp,
)
from holoscan.resources import UnboundedAllocator


class FormatPoseInferenceInputOp(Operator):
    """Operator to format input image for pose inference"""

    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        # Get input message
        in_message = op_input.receive("in")

        # Transpose
        tensor = cp.asarray(in_message.get("preprocessed_pose"))
        tensor = cp.moveaxis(tensor, 2, 0)[cp.newaxis]
        # Copy as a contiguous array to avoid issue with strides
        tensor = cp.ascontiguousarray(tensor)

        # Create output message
        op_output.emit(dict(preprocessed_pose=tensor), "out")


class PostprocessorPoseOp(Operator):
    """Operator to post-process pose inference output:
    * Non-max suppression
    * Make boxes compatible with Holoviz

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Output tensor names
        self.outputs = [
            "boxes",
            "noses",
            "left_eyes",
            "right_eyes",
            "left_ears",
            "right_ears",
            "left_shoulders",
            "right_shoulders",
            "left_elbows",
            "right_elbows",
            "left_wrists",
            "right_wrists",
            "left_hips",
            "right_hips",
            "left_knees",
            "right_knees",
            "left_ankles",
            "right_ankles",
            "segments",
        ]

        # Indices for each keypoint as defined by YOLOv8 pose model
        self.NOSE = slice(5, 7)
        self.LEFT_EYE = slice(8, 10)
        self.RIGHT_EYE = slice(11, 13)
        self.LEFT_EAR = slice(14, 16)
        self.RIGHT_EAR = slice(17, 19)
        self.LEFT_SHOULDER = slice(20, 22)
        self.RIGHT_SHOULDER = slice(23, 25)
        self.LEFT_ELBOW = slice(26, 28)
        self.RIGHT_ELBOW = slice(29, 31)
        self.LEFT_WRIST = slice(32, 34)
        self.RIGHT_WRIST = slice(35, 37)
        self.LEFT_HIP = slice(38, 40)
        self.RIGHT_HIP = slice(41, 43)
        self.LEFT_KNEE = slice(44, 46)
        self.RIGHT_KNEE = slice(47, 49)
        self.LEFT_ANKLE = slice(50, 52)
        self.RIGHT_ANKLE = slice(53, 55)

    def setup(self, spec: OperatorSpec):
        """
        input: "in"    - Input tensors coming from output of inference model
        output: "out"  - The post-processed output after applying thresholding and non-max suppression.
                         Outputs are the boxes, keypoints, and segments.  See self.outputs for the list of outputs.
        params:
            iou_threshold:    Intersection over Union (IoU) threshold for non-max suppression (default: 0.5)
            score_threshold:  Score threshold for filtering out low scores (default: 0.5)
            image_dim:        Image dimensions for normalizing the boxes (default: None)

        Returns:
            None
        """
        spec.input("in")
        spec.output("out")
        spec.param("iou_threshold", 0.5)
        spec.param("score_threshold", 0.5)
        spec.param("image_dim", None)

    def get_keypoints(self, detection):
        # Keypoints to be returned including our own "neck" keypoint
        keypoints = {
            "nose": detection[self.NOSE],
            "left_eye": detection[self.LEFT_EYE],
            "right_eye": detection[self.RIGHT_EYE],
            "left_ear": detection[self.LEFT_EAR],
            "right_ear": detection[self.RIGHT_EAR],
            "neck": (detection[self.LEFT_SHOULDER] + detection[self.RIGHT_SHOULDER]) / 2,
            "left_shoulder": detection[self.LEFT_SHOULDER],
            "right_shoulder": detection[self.RIGHT_SHOULDER],
            "left_elbow": detection[self.LEFT_ELBOW],
            "right_elbow": detection[self.RIGHT_ELBOW],
            "left_wrist": detection[self.LEFT_WRIST],
            "right_wrist": detection[self.RIGHT_WRIST],
            "left_hip": detection[self.LEFT_HIP],
            "right_hip": detection[self.RIGHT_HIP],
            "left_knee": detection[self.LEFT_KNEE],
            "right_knee": detection[self.RIGHT_KNEE],
            "left_ankle": detection[self.LEFT_ANKLE],
            "right_ankle": detection[self.RIGHT_ANKLE],
        }

        return keypoints

    def compute(self, op_input, op_output, context):
        # Get input message
        in_message = op_input.receive("in")

        # Convert input to cupy array
        results = cp.asarray(in_message.get("inference_output_pose"))[0]

        # Filter out low scores
        results = results[:, results[4, :] > self.score_threshold]
        scores = results[4, :]

        # If no detections, return zeros for all outputs
        if results.shape[1] == 0:
            out_message = Entity(context)
            zeros = hs.as_tensor(np.zeros([1, 2, 2]).astype(np.float32))

            for output in self.outputs:
                out_message.add(zeros, output)
            op_output.emit(out_message, "out")
            return

        results = results.transpose([1, 0])

        segments = []
        for i, detection in enumerate(results):
            # fmt: off
            kp = self.get_keypoints(detection)
            # Every two points defines a segment
            segments.append([kp["nose"], kp["left_eye"],      # nose <-> left eye
                             kp["nose"], kp["right_eye"],     # nose <-> right eye
                             kp["left_eye"], kp["left_ear"],  # ...
                             kp["right_eye"], kp["right_ear"],
                             kp["left_shoulder"], kp["right_shoulder"],
                             kp["left_shoulder"], kp["left_elbow"],
                             kp["left_elbow"], kp["left_wrist"],
                             kp["right_shoulder"], kp["right_elbow"],
                             kp["right_elbow"], kp["right_wrist"],
                             kp["left_shoulder"], kp["left_hip"],
                             kp["left_hip"], kp["left_knee"],
                             kp["left_knee"], kp["left_ankle"],
                             kp["right_shoulder"], kp["right_hip"],
                             kp["right_hip"], kp["right_knee"],
                             kp["right_knee"], kp["right_ankle"],
                             kp["left_hip"], kp["right_hip"],
                             kp["left_ear"], kp["neck"],
                             kp["right_ear"], kp["neck"],
                             ])
            # fmt: on

        cx, cy, w, h = results[:, 0], results[:, 1], results[:, 2], results[:, 3]
        x1, x2 = cx - w / 2, cx + w / 2
        y1, y2 = cy - h / 2, cy + h / 2

        data = {
            "boxes": cp.asarray(np.stack([x1, y1, x2, y2], axis=-1)).transpose([1, 0]),
            "noses": results[:, self.NOSE],
            "left_eyes": results[:, self.LEFT_EYE],
            "right_eyes": results[:, self.RIGHT_EYE],
            "left_ears": results[:, self.LEFT_EAR],
            "right_ears": results[:, self.RIGHT_EAR],
            "left_shoulders": results[:, self.LEFT_SHOULDER],
            "right_shoulders": results[:, self.RIGHT_SHOULDER],
            "left_elbows": results[:, self.LEFT_ELBOW],
            "right_elbows": results[:, self.RIGHT_ELBOW],
            "left_wrists": results[:, self.LEFT_WRIST],
            "right_wrists": results[:, self.RIGHT_WRIST],
            "left_hips": results[:, self.LEFT_HIP],
            "right_hips": results[:, self.RIGHT_HIP],
            "left_knees": results[:, self.LEFT_KNEE],
            "right_knees": results[:, self.RIGHT_KNEE],
            "left_ankles": results[:, self.LEFT_ANKLE],
            "right_ankles": results[:, self.RIGHT_ANKLE],
            "segments": cp.asarray(segments),
        }
        scores = cp.asarray(scores)

        out = self.nms(data, scores)

        # Rearrange boxes to be compatible with Holoviz
        out["boxes"] = cp.reshape(out["boxes"][None], (1, -1, 2))

        # Create output message
        out_message = Entity(context)
        for output in self.outputs:
            out_message.add(hs.as_tensor(out[output] / self.image_dim), output)
        op_output.emit(out_message, "out")

    def nms(self, inputs, scores):
        """Non-max suppression (NMS)
        Performs non-maximum suppression on input boxes according to their intersection-over-union (IoU).
        Filter out detections where the IoU is >= self.iou_threshold.

        See https://learnopencv.com/non-maximum-suppression-theory-and-implementation-in-pytorch/ for an introduction to non-max suppression.

        Parameters
        ----------
        inputs : dictionary containing boxes, keypoints, and segments
        scores : array (n,)

        Returns
        ----------
        outputs : dictionary containing remaining boxes, keypoints, and segments after non-max supprerssion

        """

        boxes = inputs["boxes"]
        segments = inputs["segments"]

        if len(boxes) == 0:
            return cp.asarray([]), cp.asarray([])

        # Get coordinates
        x0, y0, x1, y1 = boxes[0, :], boxes[1, :], boxes[2, :], boxes[3, :]

        # Area of bounding boxes
        area = (x1 - x0 + 1) * (y1 - y0 + 1)

        # Get indices of sorted scores
        indices = cp.argsort(scores)

        # Output boxes and scores
        boxes_out, segments_out, scores_out = [], [], []

        selected_indices = []

        # Iterate over bounding boxes
        while len(indices) > 0:
            # Get index with highest score from remaining indices
            index = indices[-1]
            selected_indices.append(index)
            # Pick bounding box with highest score
            boxes_out.append(boxes[:, index])
            segments_out.extend(segments[index])
            scores_out.append(scores[index])

            # Get coordinates
            x00 = cp.maximum(x0[index], x0[indices[:-1]])
            x11 = cp.minimum(x1[index], x1[indices[:-1]])
            y00 = cp.maximum(y0[index], y0[indices[:-1]])
            y11 = cp.minimum(y1[index], y1[indices[:-1]])

            # Compute IOU
            width = cp.maximum(0, x11 - x00 + 1)
            height = cp.maximum(0, y11 - y00 + 1)
            overlap = width * height
            union = area[index] + area[indices[:-1]] - overlap
            iou = overlap / union

            # Threshold and prune
            left = cp.where(iou < self.iou_threshold)
            indices = indices[left]

        selected_indices = cp.asarray(selected_indices)

        outputs = {
            "boxes": cp.asarray(boxes_out),
            "segments": cp.asarray(segments_out),
            "noses": inputs["noses"][selected_indices],
            "left_eyes": inputs["left_eyes"][selected_indices],
            "right_eyes": inputs["right_eyes"][selected_indices],
            "left_ears": inputs["left_ears"][selected_indices],
            "right_ears": inputs["right_ears"][selected_indices],
            "left_shoulders": inputs["left_shoulders"][selected_indices],
            "right_shoulders": inputs["right_shoulders"][selected_indices],
            "left_elbows": inputs["left_elbows"][selected_indices],
            "right_elbows": inputs["right_elbows"][selected_indices],
            "left_wrists": inputs["left_wrists"][selected_indices],
            "right_wrists": inputs["right_wrists"][selected_indices],
            "left_hips": inputs["left_hips"][selected_indices],
            "right_hips": inputs["right_hips"][selected_indices],
            "left_knees": inputs["left_knees"][selected_indices],
            "right_knees": inputs["right_knees"][selected_indices],
            "left_ankles": inputs["left_ankles"][selected_indices],
            "right_ankles": inputs["right_ankles"][selected_indices],
        }

        return outputs


class PostprocessorDepthOp(Operator):
    """Operator that does depth postprocessing before sending resulting image to Holoviz"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #
        self.image_dim = 518
        self.mouse_pressed = False
        self.display_modes = ["original", "depth", "side-by-side", "interactive"]
        self.idx = 1
        self.current_display_mode = self.display_modes[self.idx]
        # In interactive mode, how much of the original video to show
        self.ratio = 0.5

    def setup(self, spec: OperatorSpec):
        """
        input:  "input_depthmap"  - Input tensors representing depthmap from inference
        input:  "input_image"     - Input tensor representing the RGB image
        output: "output_image"    - The image for Holoviz to display
        output: "output_specs"    - Text to show the current display mode

        This operator's output image depends on the current display mode, if set to

            * "original": output the original image from input source
            * "depth": output the color depthmap based on the depthmap returned from
                       Depth Anything V2 model
            * "side-by-side": output a side-by-side view of the original image next to
                              the color depthmap
            * "interactive": allow user to control how much of the image to show as
                             original while the rest shows the color depthmap

        Returns:
            None
        """
        spec.input("input_depthmap")
        spec.input("input_image")
        spec.output("output_image")
        spec.output("output_specs")

    def clamp(self, value, min_value=0, max_value=1):
        """Clamp value between [min_value, max_value]"""
        return max(min_value, min(max_value, value))

    def toggle_display_mode(self, *args):
        mouse_button = args[0]
        action = args[1]

        LEFT_BUTTON = 0
        PRESSED = 0

        # If event is for the middle or right mouse button, update some values for interactive mode
        #   - update the status of whether the button is being pressed or released
        #   - update the ratio of the original image to display
        if mouse_button.value != LEFT_BUTTON:
            self.mouse_pressed = action.value == PRESSED
            self.x = self.clamp(self.x, 0, self.framebuffer_size)
            self.ratio = self.x / self.framebuffer_size
            return

        # When left mouse button is pressed, update the display mode
        if action.value == PRESSED:
            self.idx = (self.idx + 1) % len(self.display_modes)
            self.current_display_mode = self.display_modes[self.idx]

    # Update cursor position which will be used in interactive mode
    def cursor_pos_callback(self, *args):
        self.x = args[0]
        if self.mouse_pressed:
            self.x = self.clamp(self.x, 0, self.framebuffer_size)
            self.ratio = self.x / self.framebuffer_size

    # Update size of holoviz framer buffer which will be used to calculate self.ratio
    def framebuffer_size_callback(self, *args):
        self.framebuffer_size = args[0]

    def normalize(self, depth_map):
        min_value = cp.min(depth_map)
        max_value = cp.max(depth_map)
        normalized = (depth_map - min_value) / (max_value - min_value)
        return 255 - (normalized * 255)

    def compute(self, op_input, op_output, context):
        # Get input message
        in_message = op_input.receive("input_depthmap")
        in_image = op_input.receive("input_image")

        # Convert input to cupy array
        inference_output = cp.asarray(in_message.get("inference_output_depth")).squeeze()

        image = cp.asarray(in_image.get("preprocessed_depth"))

        if self.current_display_mode == "original":
            # Display the original image
            image = (image * 255).astype(cp.uint8)
            output_image = image
        elif self.current_display_mode == "depth":
            # Display the color depthmap
            depth_normalized = self.normalize(inference_output)
            depth_colormap = cv2.applyColorMap(
                depth_normalized.get().astype("uint8"), cv2.COLORMAP_JET
            )
            output_image = depth_colormap
        elif self.current_display_mode == "side-by-side":
            # Display both original and color depthmap images side-by-side
            depth_normalized = self.normalize(inference_output)
            depth_colormap = cv2.applyColorMap(
                depth_normalized.get().astype("uint8"), cv2.COLORMAP_JET
            )
            image = (image * 255).astype(cp.uint8)
            output_image = cp.hstack((image, depth_colormap))
        else:
            # Interactive mode
            depth_normalized = self.normalize(inference_output)
            depth_colormap = cv2.applyColorMap(
                depth_normalized.get().astype("uint8"), cv2.COLORMAP_JET
            )
            image = (image * 255).astype(cp.uint8)
            pos = int(self.image_dim * self.ratio)
            output_image = cp.hstack(
                (
                    image[:, :pos, :],
                    depth_colormap[
                        :,
                        pos:,
                    ],
                )
            )

        # Position display mode text near bottom left corner of Holoviz window
        display_mode_text = np.asarray([(0.025, 0.9)])

        # Create output message
        out_message = {"display_mode": display_mode_text, "image": hs.as_tensor(output_image)}
        op_output.emit(out_message, "output_image")

        # holoviz specs for displaying the current display mode
        specs = []
        spec = HolovizOp.InputSpec("display_mode", "text")
        spec.text = [self.current_display_mode]
        spec.color = [1.0, 1.0, 1.0, 1.0]
        spec.priority = 1
        specs.append(spec)
        op_output.emit(specs, "output_specs")


class DepthAndPoseApp(Application):
    def __init__(self, video_device="none"):
        """Initialize the depth anything v2 application"""

        super().__init__()

        self.name = "Depth And Pose App"
        self.data_path_depth = os.path.join(os.environ.get("HOLOHUB_DATA_PATH", "../data"), "depth_anything_v2")
        self.data_path_pose = os.path.join(os.environ.get("HOLOHUB_DATA_PATH", "../data"), "body_pose_estimation")
        self.video_device = video_device

    def compose(self):
        pool = UnboundedAllocator(self, name="pool")

        v4l2_args = self.kwargs("v4l2_source")
        if self.video_device != "none":
            v4l2_args["device"] = self.video_device
        source = V4L2VideoCaptureOp(
            self,
            name="v4l2_source",
            allocator=pool,
            **v4l2_args,
        )

        preprocessor_pose = FormatConverterOp(
            self,
            name="preprocessor_pose",
            pool=pool,
            **self.kwargs("preprocessor_pose"),
        )
        preprocessor_depth = FormatConverterOp(
            self,
            name="preprocessor_depth",
            pool=pool,
            **self.kwargs("preprocessor_depth"),
        )

        format_pose_input = FormatPoseInferenceInputOp(
            self,
            name="format_pose_input",
            pool=pool,
        )

        inference_args = self.kwargs("inference")
        inference_args["model_path_map"] = {
            "depth": os.path.join(self.data_path_depth, "depth_anything_v2_vits.onnx"),
            "pose": os.path.join(self.data_path_pose, "yolov8l-pose.onnx")
        }

        inference = InferenceOp(
            self,
            name="inference",
            allocator=pool,
            **inference_args,
        )

        postprocessor_pose = PostprocessorPoseOp(
            self,
            name="postprocessor_pose",
            allocator=pool,
            **self.kwargs("postprocessor_pose"),
        )
        postprocessor_depth = PostprocessorDepthOp(
            self,
            allocator=pool,
            name="postprocessor_depth"
        )

        holoviz_pose = HolovizOp(
            self,
            allocator=pool,
            name="holoviz_pose",
            window_title="Pose",
            **self.kwargs("holoviz_pose"),
        )
        holoviz_depth = HolovizOp(
            self,
            allocator=pool,
            name="holoviz_depth",
            window_title="Depth",
            mouse_button_callback=postprocessor_depth.toggle_display_mode,
            cursor_pos_callback=postprocessor_depth.cursor_pos_callback,
            framebuffer_size_callback=postprocessor_depth.framebuffer_size_callback,
            **self.kwargs("holoviz_depth")
        )

        # ==============================
        # Pose
        # ==============================
        self.add_flow(source, holoviz_pose, {("", "receivers")})
        self.add_flow(source, preprocessor_pose)
        self.add_flow(preprocessor_pose, format_pose_input)
        self.add_flow(format_pose_input, inference, {("", "receivers")})
        self.add_flow(inference, postprocessor_pose, {("transmitter", "in")})
        self.add_flow(postprocessor_pose, holoviz_pose, {("out", "receivers")})

        # ==============================
        # Depth
        # ==============================
        self.add_flow(source, preprocessor_depth)
        self.add_flow(preprocessor_depth, postprocessor_depth, {("tensor", "input_image")})
        self.add_flow(preprocessor_depth, inference, {("", "receivers")})
        self.add_flow(inference, postprocessor_depth, {("transmitter", "input_depthmap")})
        self.add_flow(postprocessor_depth, holoviz_depth, {("output_image", "receivers")})
        self.add_flow(postprocessor_depth, holoviz_depth, {("output_specs", "input_specs")})


if __name__ == "__main__":
    # Parse args
    parser = ArgumentParser(description="Depth Anything V2 Application.")
    parser.add_argument(
        "-c",
        "--config",
        default="none",
        help=("Set config path to override the default config file location"),
    )
    parser.add_argument(
        "-v",
        "--video_device",
        default="none",
        help=("The video device to use.  By default the application will use /dev/video0"),
    )

    args = parser.parse_args()

    if args.config == "none":
        config_file = os.path.join(os.path.dirname(__file__), "depth_and_pose.yaml")
    else:
        config_file = args.config

    app = DepthAndPoseApp(args.video_device)
    app.config(config_file)
    app.run()
