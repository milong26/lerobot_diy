from dataclasses import dataclass
from pathlib import Path

from ..configs import ForceSensorConfig



@ForceSensorConfig.register_subclass("WowForceSensor")
@dataclass
class WowForceSensorConfig(ForceSensorConfig):
    port: str
    mock: bool = False
    num_mags: int = 5