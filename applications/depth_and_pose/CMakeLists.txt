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
project(depth_and_pose NONE)

find_package(holoscan 1.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# ==============================
# Depth
# ==============================

# Copy depth anything v2 model if it's not in the data directory
add_custom_command(OUTPUT "${HOLOHUB_DATA_DIR}/depth_anything_v2/depth_anything_v2_vits.onnx"
    COMMAND mkdir -p ${HOLOHUB_DATA_DIR}/depth_anything_v2/
    COMMAND cp "/models/depth_anything_v2_vits.onnx" "${HOLOHUB_DATA_DIR}/depth_anything_v2/"
    BYPRODUCTS "${HOLOHUB_DATA_DIR}/depth_anything_v2/depth_anything_v2_vits.onnx"
    )

# Download the testing video and trim
add_custom_command(OUTPUT "${HOLOHUB_DATA_DIR}/depth_anything_v2/running_on_pathway.mp4"
    COMMAND mkdir -p "${HOLOHUB_DATA_DIR}/depth_anything_v2"
    COMMAND curl -S -o "${HOLOHUB_DATA_DIR}/depth_anything_v2/video.mp4"
              -L "https://www.pexels.com/download/video/5823544/?fps=50.0&h=720&w=1280"
    COMMAND ffmpeg -i "${HOLOHUB_DATA_DIR}/depth_anything_v2/video.mp4" -t 6.4
              -c:v libx264 "${HOLOHUB_DATA_DIR}/depth_anything_v2/running_on_pathway.mp4"
    COMMAND rm -rf "${HOLOHUB_DATA_DIR}/depth_anything_v2/video.mp4"
    BYPRODUCTS "${HOLOHUB_DATA_DIR}/depth_anything_v2/running_on_pathway.mp4"
    VERBATIM
    )


# Convert the video file
add_custom_command(OUTPUT "${HOLOHUB_DATA_DIR}/depth_anything_v2/running_on_pathway.gxf_index"
                          "${HOLOHUB_DATA_DIR}/depth_anything_v2/running_on_pathway.gxf_entities"
    COMMAND ffmpeg -i "${HOLOHUB_DATA_DIR}/depth_anything_v2/running_on_pathway.mp4" -pix_fmt rgb24 -f rawvideo pipe:1 |
            python3 "${CMAKE_SOURCE_DIR}/utilities/convert_video_to_gxf_entities.py"
            --directory "${HOLOHUB_DATA_DIR}/depth_anything_v2" --basename running_on_pathway --width 1280 --height 720 --framerate 50
    BYPRODUCTS "${HOLOHUB_DATA_DIR}/depth_anything_v2/running_on_pathway.gxf_index"
               "${HOLOHUB_DATA_DIR}/depth_anything_v2/running_on_pathway.gxf_entities"
    DEPENDS "${HOLOHUB_DATA_DIR}/depth_anything_v2/running_on_pathway.mp4"
    )

add_custom_target(depth_anything_v2_data ALL
                  DEPENDS
                  "${HOLOHUB_DATA_DIR}/depth_anything_v2/depth_anything_v2_vits.onnx"
                  "${HOLOHUB_DATA_DIR}/depth_anything_v2/running_on_pathway.gxf_index")

# ==============================
# Body pose
# ==============================

# Download the yolov8 pose model if it doesn't exist
add_custom_command(OUTPUT "${HOLOHUB_DATA_DIR}/body_pose_estimation/yolov8l-pose.onnx"
COMMAND "PATH=${PATH}:/workspace/holohub/.local/bin" /usr/local/bin/yolo export model=yolov8l-pose.pt format=onnx
COMMAND mkdir -p "${HOLOHUB_DATA_DIR}/body_pose_estimation"
COMMAND mv yolov8l-pose.onnx "${HOLOHUB_DATA_DIR}/body_pose_estimation"
BYPRODUCTS "${HOLOHUB_DATA_DIR}/body_pose_estimation/yolov8l-pose.onnx"
)

# Download the testing video
add_custom_command(OUTPUT "${HOLOHUB_DATA_DIR}/body_pose_estimation/twirl.mp4"
COMMAND mkdir -p "${HOLOHUB_DATA_DIR}/body_pose_estimation"
COMMAND curl -S -o "${HOLOHUB_DATA_DIR}/body_pose_estimation/twirl.mp4"
          -L "https://www.pexels.com/download/video/5385885/?fps=25.0&h=1920&w=1080"
BYPRODUCTS "${HOLOHUB_DATA_DIR}/body_pose_estimation/twirl.mp4"
VERBATIM
)

# Convert the video file
add_custom_command(OUTPUT "${HOLOHUB_DATA_DIR}/body_pose_estimation/twirl.gxf_index"
                      "${HOLOHUB_DATA_DIR}/body_pose_estimation/twirl.gxf_entities"
COMMAND ffmpeg -i "${HOLOHUB_DATA_DIR}/body_pose_estimation/twirl.mp4" -pix_fmt rgb24 -f rawvideo pipe:1 |
        python3 "${CMAKE_SOURCE_DIR}/utilities/convert_video_to_gxf_entities.py"
        --directory "${HOLOHUB_DATA_DIR}/body_pose_estimation" --basename twirl --width 1080 --height 1920 --framerate 25
BYPRODUCTS "${HOLOHUB_DATA_DIR}/body_pose_estimation/twirl.gxf_index"
           "${HOLOHUB_DATA_DIR}/body_pose_estimation/twirl.gxf_entities"
DEPENDS "${HOLOHUB_DATA_DIR}/body_pose_estimation/twirl.mp4"
)

add_custom_target(body_pose_estimation_data ALL
              DEPENDS
              "${HOLOHUB_DATA_DIR}/body_pose_estimation/yolov8l-pose.onnx"
              "${HOLOHUB_DATA_DIR}/body_pose_estimation/twirl.gxf_index")