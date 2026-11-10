"""NumPy metrics for sparse drawings; proxies expose blur and missing strokes."""

import numpy as np


def orientation_field(image, block=16):
    gy, gx = np.gradient(image)
    height, width = image.shape
    height, width = height // block * block, width // block * block

    def pool(array):
        return array[:height, :width].reshape(height // block, block, width // block, block).mean((1, 3))

    a, b, c = pool(gx * gx), pool(gy * gy), pool(gx * gy)
    angle = 0.5 * np.arctan2(2 * c, a - b)
    energy = a + b
    coherence = np.sqrt((a - b) ** 2 + 4 * c * c) / np.maximum(energy, 1e-8)
    return angle, energy, coherence


def metrics(prediction, target, features):
    contour = features[6] > 0
    support = (features[1] > 0) | (features[8] > 0)
    inside = support & ~contour
    result = {"raster_mae": float(np.mean(np.abs(prediction - target))),
              "white_baseline_mae": float(np.mean(1 - target)),
              "contour_mae": float(np.abs(prediction - target)[contour].mean()) if contour.any() else None,
              "predicted_ink_density": float((1 - prediction)[inside].mean()) if inside.any() else None,
              "teacher_ink_density": float((1 - target)[inside].mean()) if inside.any() else None}
    pg_y, pg_x = np.gradient(prediction)
    tg_y, tg_x = np.gradient(target)
    p_edge = np.hypot(pg_x, pg_y)
    t_edge = np.hypot(tg_x, tg_y)
    # Area / edge contrast is a width proxy for isolated constant-opacity strokes.
    # Smooth wash, overlap, and varying opacity bias it, so report contrast too.
    for name, image, edge in [("predicted", prediction, p_edge), ("teacher", target, t_edge)]:
        edge_sum = float(edge[inside].sum())
        result[f"{name}_width_proxy_px"] = float(2 * (1 - image)[inside].sum() / edge_sum) if edge_sum > 1e-6 else None
        result[f"{name}_edge_contrast"] = float(edge[inside].mean()) if inside.any() else None
    block = 8
    h, w = prediction.shape

    def pool(image):
        return image.reshape(h // block, block, w // block, block).mean((1, 3))

    result["shadow_tone_mae"] = float(np.abs(pool(prediction) - pool(target)).mean())
    pa, pe, pc = orientation_field(prediction)
    ta, te, tc = orientation_field(target)
    reliable_target = (te > 0.0005) & (tc > 0.25)
    def patch_mean(array):
        return array.reshape(h // 16, 16, w // 16, 16).mean((1, 3))

    reliable_target &= (patch_mean(contour) < 0.10) & (patch_mean(support) > 0.5)
    reliable = reliable_target & (pe > 0.0005) & (pc > 0.25)
    angle_error = np.rad2deg(np.arccos(np.clip(np.abs(np.cos(pa - ta)), 0, 1)))
    result["stroke_angle_error_degrees"] = float(angle_error[reliable].mean()) if reliable.any() else None
    result["orientation_coverage"] = float(reliable.sum() / max(reliable_target.sum(), 1))
    return result


