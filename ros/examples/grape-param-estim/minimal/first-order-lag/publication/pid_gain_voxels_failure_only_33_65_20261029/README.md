# Failure-only PID maps: 33 and 65 grids

Both failure recordings are evaluated independently, with all original plant samples in each denominator. Other nine gains remain at that same recording's values.

The figures follow the earlier publication layout and styles; the withdrawn success-conditioned statistic is not used. No cross-failure fusion has been performed.

`data/` contains final coordinate axes and stable/valid/unresolved counts, sample coordinates, fixed controller context and source metadata. `sample_bits.npz` losslessly preserves every plant's stable/valid bit in original sample order (little-endian bit packing), plus evaluated masks. Large per-plant spectral radii and checkpoints remain locally at paths and SHA-256 hashes in `local_data_catalog.json`.

Original and runtime estimate files retain their exact numerical values. The runtime input still points to its local frozen model; a byte-identical `inputs/vehicle_model.json` is included for portability.

See `figures/grid_33/` and `figures/grid_65/` for the PDF, SVG and PNG panel figures.

The [October 30 color-rule comparison](COLOR_RULE_AUDIT_20261030.md) explains why the corrected numerical fractions change the colors. The continuous 0–100% palette and opacity are retained; its historical 8-bit color-anchor rounding is now reproduced exactly. Both resolutions have been redrawn from unchanged final data. [Machine-readable comparisons](color_rule_audit_20261030.json) include all numerical-file hashes and the exact palette comparison.

The figures now also show blue diamonds labeled `Success ref.`, the gain coordinates and `rate`, using the recorded successful gains in [successful_gain_reference.json](successful_gain_reference.json). The blue rate is the independently evaluated full successful controller (all 12 successful gains), on all original samples of the corresponding failure recording. The colored background still fixes the other nine gains to that failure recording; its value at the blue coordinates is a different quantity. The independent results and every per-sample spectral radius are saved in [successful_controller_comparison.json](successful_controller_comparison.json). Purple marks remain the actual full failure controller. At Z the two coordinate references coincide and are drawn as a blue diamond within a purple ring.

Roll/Pitch's successful D gain is 20, beyond the computed grid's upper D gain of 16. Only the display frame is enlarged to show this reference. The extra area is uncolored; grid coordinates, fractions, samples, and fixed gains are unchanged. See [SUCCESS_GAIN_OVERLAY_20261030.md](SUCCESS_GAIN_OVERLAY_20261030.md) for the reference and reproduction command.

Each annotation states its fixed-gain context directly: `rate (rest gains: failure)` for the purple failure mark, and `rate (rest gains: success)` for the blue success reference. Here `rest gains` means the nine gains outside the displayed P/I/D triple. The existing 33/65 PDF, SVG and PNG files are updated in place; the rates and background data are unchanged.
