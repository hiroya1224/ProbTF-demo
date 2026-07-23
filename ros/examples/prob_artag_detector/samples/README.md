# AprilTag 36h11 samples

All PNG files in this directory use OpenCV's `DICT_APRILTAG_36h11`
dictionary, which is the detector's default `family`.

| File | Marker ID | Used by |
|---|---:|---|
| `apriltag_36h11_id_000.png` | 0 | general sample |
| `apriltag_36h11_id_007.png` | 7 | pose-mixture tests |
| `apriltag_36h11_id_017.png` | 17 | detector compatibility test |
| `apriltag_36h11_id_021.png` | 21 | real-camera integration smoke test |

Each image is 2000 by 2000 pixels. The encoded black-border marker occupies
the centered 1600 by 1600 pixels, with a one-module white quiet zone on every
side. Do not crop, mirror, invert, or apply smoothing to the image.

## Display on a monitor

Open one PNG at a time and keep the complete white margin visible. Make the tag
large enough that one of its eight modules spans several camera pixels. Avoid
screen glare and extreme viewing angles.

## Print at the demo's default size

The detector's default `tag_size_m` is `0.12`, and it refers to the outside
width of the **black marker square**, excluding the white quiet zone. To obtain
a 120 mm marker from these PNGs, print the complete 2000-pixel image at 150 mm
wide:

```text
150 mm complete image width × (1600 / 2000) = 120 mm black marker width
```

Disable “fit to page” and verify the black square with a ruler. If you print a
different black-square width, set `tag_size_m` to that measured value in metres.
An incorrect size does not normally prevent ID detection, but it scales the
estimated translation.

## Regeneration

The committed PNGs can be regenerated with the package's stock OpenCV:

```bash
PYTHONNOUSERSITE=1 /usr/bin/python3 samples/generate_sample_tags.py
```

The script selects the legacy or current OpenCV ArUco API automatically.
