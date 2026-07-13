"""Visualization-only helpers."""


def pack_rgb(red, green, blue):
    return (int(red) << 16) | (int(green) << 8) | int(blue)
