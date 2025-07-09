#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import platform
from pathlib import Path
from typing import TypeAlias

from lerobot.forcesensors.configs import ForceSensorConfig
from lerobot.forcesensors.forcesensor import ForceSensor




IndexOrPath: TypeAlias = int | Path

def make_force_sensor_from_configs(forcesensor_configs: dict[str, ForceSensorConfig]) -> dict[str, ForceSensor]:
    force_sensors = {}

    for key, cfg in forcesensor_configs.items():
        if cfg.type == "WowForceSensor":
            from .WowSkin import WowForceSensor
            force_sensors[key] = WowForceSensor(cfg)
        else:
            raise ValueError(f"The motor type '{cfg.type}' is not valid.")

    return force_sensors