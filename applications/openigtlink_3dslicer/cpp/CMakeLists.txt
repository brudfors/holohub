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

cmake_minimum_required(VERSION 3.20)
project(openigtlink_3dslicer LANGUAGES CXX)

find_package(holoscan 1.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_executable(openigtlink_3dslicer
  main.cpp
)

target_link_libraries(openigtlink_3dslicer
  PRIVATE
  holoscan::core
  holoscan::ops::video_stream_replayer
  holoscan::ops::format_converter
  holoscan::ops::inference
  holoscan::ops::segmentation_postprocessor
  holoscan::ops::holoviz
  holoscan::ops::openigtlink_tx
  holoscan::ops::openigtlink_rx
)

# Download the associated dataset if needed
option(HOLOHUB_DOWNLOAD_DATASETS "Download datasets" ON)
if(HOLOHUB_DOWNLOAD_DATASETS)
  add_dependencies(openigtlink_3dslicer ultrasound_segmentation_data)
endif()

add_custom_target(openigtlink_3dslicer_yaml
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/openigtlink_3dslicer.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "openigtlink_3dslicer.yaml"
  BYPRODUCTS "openigtlink_3dslicer.yaml"
)
add_dependencies(openigtlink_3dslicer openigtlink_3dslicer_yaml)