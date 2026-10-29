# Failure-only PID maps: 33 and 65 grids

Both failure recordings are evaluated independently, with all original plant samples in each denominator. Other nine gains remain at that same recording's values.

The figures follow the earlier publication layout and styles; the withdrawn success-conditioned statistic is not used. No cross-failure fusion has been performed.

`data/` contains final coordinate axes and stable/valid/unresolved counts, sample coordinates, fixed controller context and source metadata. `sample_bits.npz` losslessly preserves every plant's stable/valid bit in original sample order (little-endian bit packing), plus evaluated masks. Large per-plant spectral radii and checkpoints remain locally at paths and SHA-256 hashes in `local_data_catalog.json`.

Original and runtime estimate files retain their exact numerical values. The runtime input still points to its local frozen model; a byte-identical `inputs/vehicle_model.json` is included for portability.

See `figures/grid_33/` and `figures/grid_65/` for the PDF, SVG and PNG panel figures.
