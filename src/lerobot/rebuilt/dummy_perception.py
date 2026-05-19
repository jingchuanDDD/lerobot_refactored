"""DummyPerception — no-op PerceptionModule for simulation without cameras."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from lerobot.rebuilt.perception.core import Observation, PerceptionModule

logger = logging.getLogger(__name__)


class DummyPerception(PerceptionModule):
    """PerceptionModule that returns empty observations.

    Use when cameras are not available (pure simulation, headless).
    """

    def __init__(self):
        self._connected = False

    @property
    def name(self) -> str:
        return "dummy_perception"

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def observation_features(self) -> dict[str, tuple]:
        return {}

    def connect(self) -> None:
        self._connected = True
        logger.info("DummyPerception connected (no cameras)")

    def disconnect(self) -> None:
        self._connected = False

    def get_observation(self) -> Observation:
        return Observation(images={})
