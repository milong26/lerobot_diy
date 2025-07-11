
import abc
from typing import Any, Dict, List

import numpy as np

from .configs import ForceSensorConfig

# sensor的base clas


class ForceSensor(abc.ABC):
    """Base class for sensor implementations.
    """

    def __init__(self, config: ForceSensorConfig):
        """Initialize the sensor with the given configuration.

        Args:
            config: sensor configuration containing FPS and resolution.
        """

    # @property
    # @abc.abstractmethod
    # def is_connected(self) -> bool:
    #     """Check if the sensor is currently connected.

    #     Returns:
    #         bool: True if the sensor is connected and ready to capture frames,
    #               False otherwise.
    #     """
    #     pass

    # @staticmethod
    # @abc.abstractmethod
    # def find_sensors() -> List[Dict[str, Any]]:
    #     """Detects available sensors connected to the system.
    #     Returns:
    #         List[Dict[str, Any]]: A list of dictionaries,
    #         where each dictionary contains information about a detected sensor.
    #     """
    #     pass

    @abc.abstractmethod
    def connect(self) -> None:
        """Establish connection to the sensor.

        Args:
            warmup: If True (default), captures a warmup frame before returning. Useful
                   for sensors that require time to adjust capture settings.
                   If False, skips the warmup frame.
        """
        pass

    # @abc.abstractmethod
    # def read(self) -> np.ndarray:
    #     """Capture and return a single frame from the sensor.

    #     Args:
    #         color_mode: Desired color mode for the output frame. If None,
    #                     uses the sensor's default color mode.

    #     Returns:
    #         np.ndarray: Captured frame as a numpy array.
    #     """
    #     pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the sensor and release resources."""
        pass
