"""Map the supported synthetic-scene prompt vocabulary to CPU model labels.

This is a small deterministic adapter for the model's closed curriculum. It is
not a general-purpose language encoder; unsupported multi-object and event
prompts are rejected so the sampler cannot silently return the wrong scene.
"""

from __future__ import annotations

import re


_SHAPES = {"circle": 0, "square": 1, "triangle": 2}
_COLORS = {"red": 0, "green": 1, "blue": 2, "yellow": 3}
_SIZES = {"small": "small", "medium": "medium", "large": "large"}
_MOTIONS = {
    "left to right": "left_to_right",
    "right to left": "right_to_left",
    "top to bottom": "top_to_bottom",
    "bottom to top": "bottom_to_top",
    "diagonal down right": "diagonal_down_right",
    "diagonal down left": "diagonal_down_left",
    "stationary": "stationary",
}


def _find_unique(text: str, mapping: dict[str, object], label: str):
    found = [value for word, value in mapping.items()
             if re.search(rf"\b{re.escape(word)}s?\b", text)]
    unique = list(dict.fromkeys(found))
    if len(unique) > 1:
        raise ValueError(f"Prompt specifies multiple {label} values")
    return unique[0] if unique else None


def parse_scene_prompt(prompt: str) -> dict:
    """Extract one shape, palette color, optional size, motion, and speed."""
    text = re.sub(r"(?<=[a-z])[-_](?=[a-z])", " ", " ".join(prompt.lower().strip().split()))
    if not text:
        raise ValueError("Prompt cannot be empty")
    if re.search(r"\b(two|2|three|3|four|4|multiple|several|pair of)\b", text):
        raise ValueError("This CPU curriculum supports one object per prompt")
    if re.search(r"\b(collid\w*|bounc\w*|rotat\w*|spinn?\w*|jump\w*|fall\w*)\b", text):
        raise ValueError("Collision, bounce, rotation, and other event prompts are not supported by this sampler")

    shape = _find_unique(text, _SHAPES, "shape")
    color = _find_unique(text, _COLORS, "color")
    if shape is None or color is None:
        missing = "shape" if shape is None else "color"
        raise ValueError(f"Prompt must name one supported {missing}: {', '.join(_SHAPES if missing == 'shape' else _COLORS)}")

    size = _find_unique(text, _SIZES, "size")
    motion = None
    for phrase, label in _MOTIONS.items():
        if re.search(rf"\b{re.escape(phrase)}\b", text):
            if motion is not None and motion != label:
                raise ValueError("Prompt specifies multiple motion directions")
            motion = label
    if motion is None:
        if re.search(r"\b(move|moves|moving|travel|travels|traveling|go|goes)\b", text):
            motion = "left_to_right"
        else:
            motion = "stationary"

    has_slow = bool(re.search(r"\b(slow|slowly|gently)\b", text))
    has_fast = bool(re.search(r"\b(fast|quick|quickly)\b", text))
    if has_slow and has_fast:
        raise ValueError("Prompt specifies conflicting speed values")
    speed = "slowly" if has_slow else "quickly"
    return {"shape": shape, "color": color, "size": size, "motion": motion, "speed": speed}
