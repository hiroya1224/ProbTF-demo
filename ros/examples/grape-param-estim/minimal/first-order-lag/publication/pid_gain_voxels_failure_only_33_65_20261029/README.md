# Failure-only PID maps: 33 and 65 grids

Both failure recordings are evaluated independently, with all original plant samples in each denominator. Other nine gains remain at that same recording's values.

The figures follow the earlier publication layout and styles; the withdrawn success-conditioned statistic is not used. No cross-failure fusion has been performed.

`data/` contains final coordinate axes and stable/valid/unresolved counts, sample coordinates, fixed controller context and source metadata. `sample_bits.npz` losslessly preserves every plant's stable/valid bit in original sample order (little-endian bit packing), plus evaluated masks. Large per-plant spectral radii and checkpoints remain locally at paths and SHA-256 hashes in `local_data_catalog.json`.

Original and runtime estimate files retain their exact numerical values. The runtime input still points to its local frozen model; a byte-identical `inputs/vehicle_model.json` is included for portability.

See `figures/grid_33/` and `figures/grid_65/` for the PDF, SVG and PNG panel figures.

The [October 30 color-rule comparison](COLOR_RULE_AUDIT_20261030.md) explains why the corrected numerical fractions change the colors. The continuous 0–100% palette and opacity are retained; its historical 8-bit color-anchor rounding is now reproduced exactly. Both resolutions have been redrawn from unchanged final data. [Machine-readable comparisons](color_rule_audit_20261030.json) include all numerical-file hashes and the exact palette comparison.

The figures show the recorded failure triple (purple) and successful triple (blue), using the successful coordinates in [successful_gain_reference.json](successful_gain_reference.json). **Both points carry both remaining-gain contexts:** under a centered `rest PID` heading, `success: rate = ...` fixes the other nine gains to the successful values, and `failure: rate = ...` fixes them to that failure recording. The point's own displayed P/I/D triple stays fixed in both rows. All original plant samples and the same denominator are used for every comparison. The colored background remains the failure context. At Z the two coordinate references coincide and are drawn as a blue diamond within a purple ring.

Roll/Pitch's successful D gain is 20, beyond the computed grid's upper D gain of 16. Only the display frame is enlarged to show this reference. The extra area is uncolored; grid coordinates, fractions, samples, and fixed gains are unchanged. See [SUCCESS_GAIN_OVERLAY_20261030.md](SUCCESS_GAIN_OVERLAY_20261030.md) for the reference and reproduction command.

The paired point comparisons and every per-sample spectral radius are saved in [rest_pid_comparison.json](rest_pid_comparison.json). The earlier full-successful-controller subset remains in [successful_controller_comparison.json](successful_controller_comparison.json). The existing 33/65 PDF, SVG and PNG files are updated in place. No plant selection, success conditioning, or modification of the numerical background maps is performed.
