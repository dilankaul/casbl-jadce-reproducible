from __future__ import annotations
from pathlib import Path
import numpy as np
from .activity_model import ActivitySample
from .io import ensure_dir


def save_activity_samples(path: str | Path, samples: list[ActivitySample]) -> None:
    p = Path(path); ensure_dir(p.parent)
    np.savez_compressed(
        p,
        a=np.stack([s.a for s in samples]),
        device_locations=np.stack([s.device_locations for s in samples]),
        event_locations=np.stack([s.event_locations for s in samples]),
        activation_probabilities=np.stack([s.activation_probabilities for s in samples]),
    )


def load_activity_samples(path: str | Path) -> list[ActivitySample]:
    data = np.load(path)
    return [
        ActivitySample(
            a=data["a"][i].astype(bool),
            device_locations=data["device_locations"][i],
            event_locations=data["event_locations"][i],
            activation_probabilities=data["activation_probabilities"][i],
        )
        for i in range(data["a"].shape[0])
    ]
