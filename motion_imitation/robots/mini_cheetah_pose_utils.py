# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions to calculate Mini Cheetah's pose and motor angles."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import attr

DEFAULT_ABDUCTION_ANGLE = 0
DEFAULT_HIP_ANGLE = -.8
DEFAULT_KNEE_ANGLE = 1.6

@attr.s
class MiniCheetahPose(object):
  """Default pose of the MiniCheeah.

    Leg order:
    0 -> Left Front.
    1 -> Left Hind.
    2 -> Right Front.
    3 -> Right Hind.
  """
  abduction_angle_0 = attr.ib(type=float, default=0)
  hip_angle_0 = attr.ib(type=float, default=0)
  knee_angle_0 = attr.ib(type=float, default=0)
  abduction_angle_1 = attr.ib(type=float, default=0)
  hip_angle_1 = attr.ib(type=float, default=0)
  knee_angle_1 = attr.ib(type=float, default=0)
  abduction_angle_2 = attr.ib(type=float, default=0)
  hip_angle_2 = attr.ib(type=float, default=0)
  knee_angle_2 = attr.ib(type=float, default=0)
  abduction_angle_3 = attr.ib(type=float, default=0)
  hip_angle_3 = attr.ib(type=float, default=0)
  knee_angle_3 = attr.ib(type=float, default=0)
