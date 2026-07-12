import csv
import json
import math
from pathlib import Path

import numpy as np


CHI2_99_DF3_SQRT = 3.3682141752187276


def covariance_ellipsoid(mean, cov, scale=1.0, resolution=28):
    center = np.asarray(mean, dtype=float).reshape(3)
    covariance = 0.5 * (np.asarray(cov, dtype=float).reshape(3, 3) + np.asarray(cov, dtype=float).reshape(3, 3).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    radii = scale * np.sqrt(np.maximum(eigenvalues, 0.0))

    u_values = np.linspace(0.0, 2.0 * math.pi, resolution)
    v_values = np.linspace(0.0, math.pi, resolution)
    x_values = np.outer(np.cos(u_values), np.sin(v_values))
    y_values = np.outer(np.sin(u_values), np.sin(v_values))
    z_values = np.outer(np.ones_like(u_values), np.cos(v_values))
    sphere = np.stack([x_values, y_values, z_values], axis=0).reshape(3, -1)

    ellipsoid = eigenvectors @ np.diag(radii) @ sphere
    ellipsoid = ellipsoid.reshape(3, resolution, resolution)
    return ellipsoid[0] + center[0], ellipsoid[1] + center[1], ellipsoid[2] + center[2]


def plot_link_prob_tf(results, out_png):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("matplotlib is required to render link_prob_tf.png.") from exc

    output_path = Path(out_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(9.0, 7.0))
    axis = figure.add_subplot(111, projection="3d")

    chain_points = [np.zeros(3, dtype=float)]
    for result in results:
        chain_points.append(np.asarray(result.mean_translation, dtype=float))
    chain_points = np.asarray(chain_points, dtype=float)

    axis.plot(chain_points[:, 0], chain_points[:, 1], chain_points[:, 2], color="black", linewidth=2.0)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(results), 1)))
    for color, result in zip(colors, results):
        mean = np.asarray(result.mean_translation, dtype=float)
        cov = np.asarray(result.cov_translation, dtype=float)
        ellipsoid = covariance_ellipsoid(mean, cov, scale=CHI2_99_DF3_SQRT)
        axis.plot_surface(
            ellipsoid[0],
            ellipsoid[1],
            ellipsoid[2],
            rstride=1,
            cstride=1,
            color=color,
            alpha=0.18,
            linewidth=0.0,
            shade=False,
        )
        axis.scatter([mean[0]], [mean[1]], [mean[2]], color=color, s=28)
        axis.text(mean[0], mean[1], mean[2], result.target, fontsize=8)

    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("z [m]")
    axis.set_title("Prob-TF link-origin means and covariance ellipsoids")
    axis.set_box_aspect(
        [
            max(np.ptp(chain_points[:, 0]), 0.3),
            max(np.ptp(chain_points[:, 1]), 0.3),
            max(np.ptp(chain_points[:, 2]), 0.3),
        ]
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_results_csv(results, out_csv):
    output_path = Path(out_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame",
                "mean_x",
                "mean_y",
                "mean_z",
                "cov_xx",
                "cov_xy",
                "cov_xz",
                "cov_yy",
                "cov_yz",
                "cov_zz",
                "std_x",
                "std_y",
                "std_z",
                "trace_cov",
            ]
        )
        for result in results:
            mean = np.asarray(result.mean_translation, dtype=float)
            cov = np.asarray(result.cov_translation, dtype=float)
            std = np.sqrt(np.maximum(np.diag(cov), 0.0))
            writer.writerow(
                [
                    result.target,
                    float(mean[0]),
                    float(mean[1]),
                    float(mean[2]),
                    float(cov[0, 0]),
                    float(cov[0, 1]),
                    float(cov[0, 2]),
                    float(cov[1, 1]),
                    float(cov[1, 2]),
                    float(cov[2, 2]),
                    float(std[0]),
                    float(std[1]),
                    float(std[2]),
                    float(np.trace(cov)),
                ]
            )


def write_results_json(results, out_json):
    output_path = Path(out_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = []
    for result in results:
        payload.append(
            {
                "source": result.source,
                "target": result.target,
                "mean_translation": np.asarray(result.mean_translation, dtype=float).tolist(),
                "cov_translation": np.asarray(result.cov_translation, dtype=float).tolist(),
                "mean_rotation": None if result.mean_rotation is None else np.asarray(result.mean_rotation, dtype=float).tolist(),
                "bingham_rotation": None
                if result.bingham_rotation is None
                else np.asarray(result.bingham_rotation, dtype=float).tolist(),
                "path": [f"{view.edge_id}:{view.direction:+d}" for view in result.path] if result.path is not None else [],
                "method": result.method,
                "closure_approximation": bool(result.closure_approximation),
            }
        )

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
