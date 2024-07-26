# syntax=docker/dockerfile:1

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

ARG BASE_IMAGE
ARG GPU_TYPE
ARG COMPUTE_CAPACITY
############################################################
# Torch image (Used to extract the ARM64 dGPU pytorch wheel)
############################################################
FROM nvcr.io/nvidia/pytorch:23.10-py3 as pytorch

############################################################
# Base image
############################################################
FROM ${BASE_IMAGE} as base
ARG DEBIAN_FRONTEND=noninteractive

# Holohub setup
RUN mkdir -p /tmp/scripts
COPY run /tmp/scripts/
RUN chmod +x /tmp/scripts/run
RUN /tmp/scripts/run setup

# Copy over app-specific requirements
COPY applications/ultrasound_segmentation/python/requirements.txt /tmp/requirements.txt

# Install setuptools prior to all other requirements to avoid install errors
RUN python3 -m pip install --no-cache-dir setuptools && \
    python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

# Install QT dependencies
RUN apt-get install -y \
    qt6-base-dev \
    libxcb-cursor0

WORKDIR /workspace/holohub
