"""The thresholds the page changes: kept in `<models folder>/settings.json`, so they outlive the container.

TAGGER_THRESHOLD and TAGGER_CHARACTER_THRESHOLD are the values before anything is saved. A request may still name
its own with `X-Threshold` / `X-Character-Threshold`.
"""

from __future__ import annotations

import json
import os
import threading

FILE = "settings.json"


class Preferences:
    def __init__(self, folder: str, threshold: float, character_threshold: float) -> None:
        self._path = os.path.join(folder, FILE)
        self._lock = threading.Lock()
        self._values = {"threshold": threshold, "characterThreshold": character_threshold}
        try:
            with open(self._path, encoding="utf-8") as file:
                saved = json.load(file)
            for key in self._values:
                if isinstance(saved, dict) and _valid(saved.get(key)):
                    self._values[key] = float(saved[key])
        except (OSError, ValueError):
            pass  # nothing saved yet, or a broken file: the variables' values

    def get(self) -> dict[str, float]:
        with self._lock:
            return dict(self._values)

    def update(self, changes: dict) -> dict[str, float]:
        """Sets the given members (0..1 each, else ValueError naming the member) and saves them."""
        unknown = set(changes) - set(self._values)
        if unknown:
            raise ValueError(f"Unknown setting: {', '.join(sorted(unknown))}.")
        for key, value in changes.items():
            if not _valid(value):
                raise ValueError(f"{key} must be a number from 0 to 1.")
        with self._lock:
            values = {**self._values, **{key: float(value) for key, value in changes.items()}}
            os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
            with open(self._path + ".part", "w", encoding="utf-8") as file:
                json.dump(values, file, indent=2)
            os.replace(self._path + ".part", self._path)
            self._values = values
            return dict(values)


def _valid(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1
