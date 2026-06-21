from dataclasses import dataclass


@dataclass
class Layer:
    id: int
    name: str
    masks: list
    frames: list
