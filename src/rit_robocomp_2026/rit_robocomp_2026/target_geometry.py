import math

import numpy as np


def median_depth_metres(depth_image, u, v, radius=3):
    height, width = depth_image.shape[:2]

    u0 = max(0, int(u) - radius)
    u1 = min(width, int(u) + radius + 1)
    v0 = max(0, int(v) - radius)
    v1 = min(height, int(v) + radius + 1)

    values = np.asarray(
        depth_image[v0:v1, u0:u1],
        dtype=float
    ).reshape(-1)

    values = values[
        np.isfinite(values)
        & (values > 0.0)
    ]

    if values.size == 0:
        return None

    depth = float(np.median(values))

    if depth > 20.0:
        depth /= 1000.0

    if not math.isfinite(depth) or depth <= 0.0:
        return None

    return depth


def foreground_depth_metres(
    depth_image,
    x,
    y,
    width,
    height,
):
    image_height, image_width = depth_image.shape[:2]

    margin_x = max(
        1,
        int(round(width * 0.15))
    )

    margin_y = max(
        1,
        int(round(height * 0.15))
    )

    x0 = max(
        0,
        int(x) + margin_x
    )

    x1 = min(
        image_width,
        int(x + width) - margin_x
    )

    y0 = max(
        0,
        int(y) + margin_y
    )

    y1 = min(
        image_height,
        int(y + height) - margin_y
    )

    if x1 <= x0 or y1 <= y0:
        return None

    values = np.asarray(
        depth_image[y0:y1, x0:x1],
        dtype=float
    ).reshape(-1)

    values = values[
        np.isfinite(values)
        & (values > 0.0)
    ]

    if values.size == 0:
        return None

    if float(np.median(values)) > 20.0:
        values = values / 1000.0

    near_depth = float(
        np.percentile(values, 30.0)
    )

    foreground = values[
        values <= near_depth + 0.04
    ]

    if foreground.size == 0:
        return None

    depth = float(
        np.median(foreground)
    )

    if not math.isfinite(depth) or depth <= 0.0:
        return None

    return depth


def pixel_to_optical_point(
    u,
    v,
    depth,
    fx,
    fy,
    cx,
    cy,
):
    if depth is None or fx <= 0.0 or fy <= 0.0:
        return None

    return (
        (float(u) - cx) * depth / fx,
        (float(v) - cy) * depth / fy,
        depth,
    )
