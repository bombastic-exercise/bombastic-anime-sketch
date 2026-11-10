"""Analytic ellipsoid scenes with orthographic cameras and ray-traced shadows.

These are training scaffolds, not character assets. World Y is up. The receiver
plane is y=-1.05; light vectors point from a surface toward the source.
"""

from dataclasses import asdict, dataclass

import numpy as np

SHAPES = ("sphere", "cylinder", "plate", "helmet", "arm")


@dataclass(frozen=True)
class Scene:
    shape: str = "helmet"
    geometry_seed: int = 0
    camera_yaw: float = 15.0
    camera_elevation: float = 12.0
    light_azimuth: float = -45.0
    light_elevation: float = 45.0
    material: float = 0.3

    def __post_init__(self):
        if self.shape not in SHAPES:
            raise ValueError(f"Unknown shape: {self.shape}; choose from {SHAPES}")
        if not -180 <= self.camera_yaw <= 180 or not 5 <= self.camera_elevation <= 70:
            raise ValueError("Camera yaw must be [-180,180] and elevation [5,70] degrees")
        if not -180 <= self.light_azimuth <= 180 or not 5 <= self.light_elevation <= 85:
            raise ValueError("Light azimuth must be [-180,180] and elevation [5,85] degrees")
        if not 0 <= self.material <= 1:
            raise ValueError("material must be in [0,1]")

    def to_dict(self):
        return asdict(self)


def rotation(yaw=0.0, pitch=0.0, roll=0.0):
    y, p, r = np.deg2rad([yaw, pitch, roll])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return (ry @ rx @ rz).astype(np.float32)


def primitives(scene):
    rng = np.random.default_rng(scene.geometry_seed)
    scale = rng.uniform(0.88, 1.08, 3)
    turn = rng.uniform(-15, 15)
    parts = []

    def add(center, radii, yaw=0, pitch=0, roll=0):
        parts.append((np.array(center, dtype=np.float32) * scale,
                      np.array(radii, dtype=np.float32) * scale,
                      rotation(yaw + turn, pitch, roll)))

    if scene.shape == "sphere":
        add((0, 0, 0), (0.72, 0.78, 0.72))
    elif scene.shape == "cylinder":
        # An elongated rounded volume, not a flat-ended engineering cylinder.
        add((0, 0, 0), (0.44, 0.94, 0.44), roll=12)
    elif scene.shape == "plate":
        add((0, 0, 0), (0.85, 0.7, 0.20), yaw=20, roll=-12)
        add((0, 0.55, 0.10), (0.76, 0.10, 0.21), yaw=20, roll=-12)
    elif scene.shape == "helmet":
        add((0, 0.08, 0), (0.59, 0.72, 0.5))
        add((0, 0.14, 0.42), (0.68, 0.10, 0.16))
        add((0, -0.28, 0.42), (0.17, 0.39, 0.12))
        for side in (-1, 1):
            add((side * 0.68, 0.47, 0), (0.18, 0.52, 0.10), roll=-side * 38)
    else:
        add((-0.23, 0.40, 0), (0.27, 0.58, 0.27), roll=22)
        add((0.04, -0.08, 0.06), (0.29, 0.27, 0.29))
        add((0.27, -0.48, 0.09), (0.23, 0.49, 0.24), roll=-25)
        for y in (-0.38, -0.58):
            add((0.3, y, 0.22), (0.26, 0.07, 0.15), roll=-25)
    return parts


def intersections(origin, direction, parts):
    distance = np.full(origin.shape[:-1], np.inf, dtype=np.float32)
    index = np.full(origin.shape[:-1], -1, dtype=np.int32)
    for i, (center, radii, matrix) in enumerate(parts):
        o = ((origin - center) @ matrix) / radii
        d = (direction @ matrix) / radii
        a = np.sum(d * d, axis=-1)
        b = np.sum(o * d, axis=-1)
        c = np.sum(o * o, axis=-1) - 1
        discriminant = b * b - a * c
        root = np.sqrt(np.maximum(discriminant, 0))
        near, far = (-b - root) / a, (-b + root) / a
        t = np.where(near > 1e-4, near, far)
        valid = (discriminant >= 0) & (t > 1e-4) & (t < distance)
        distance = np.where(valid, t, distance)
        index = np.where(valid, i, index)
    return distance, index


def boundaries(values):
    padded = np.pad(values, 1, mode="edge")
    return ((padded[1:-1, 1:-1] != padded[:-2, 1:-1]) |
            (padded[1:-1, 1:-1] != padded[2:, 1:-1]) |
            (padded[1:-1, 1:-1] != padded[1:-1, :-2]) |
            (padded[1:-1, 1:-1] != padded[1:-1, 2:])).astype(np.float32)


def render_geometry(scene, size=256):
    if size < 32 or size > 1024 or size % 8:
        raise ValueError("Image size must be a multiple of 8 between 32 and 1024")
    matrix = rotation(scene.camera_yaw, -scene.camera_elevation)
    axis = np.linspace(-1.5, 1.5, size, dtype=np.float32)
    x, y = np.meshgrid(axis, -axis)
    origins = np.stack([x, y, np.full_like(x, 4)], -1) @ matrix.T
    direction = np.array([0, 0, -1], np.float32) @ matrix.T
    parts = primitives(scene)
    distance, index = intersections(origins, direction, parts)
    plane_t = (-1.05 - origins[..., 1]) / direction[1]
    plane = (plane_t > 0) & (plane_t < distance)
    distance = np.where(plane, plane_t, distance)
    index = np.where(plane, -2, index)
    valid = np.isfinite(distance)
    points = origins + np.where(valid, distance, 0)[..., None] * direction
    normals = np.zeros_like(points)
    normals[plane] = (0, 1, 0)
    u, v = points[..., 0].copy(), points[..., 2].copy()
    for i, (center, radii, part_rotation) in enumerate(parts):
        selected = index == i
        local = (points[selected] - center) @ part_rotation
        n = (local / (radii * radii)) @ part_rotation.T
        normals[selected] = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)
        # Smooth local chart on the visible front; a seam exists on the back.
        u[selected] = np.arctan2(local[:, 0] / radii[0], local[:, 2] / radii[2]) * radii[0]
        v[selected] = local[:, 1]
    az, el = np.deg2rad([scene.light_azimuth, scene.light_elevation])
    light = np.array([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)], np.float32)
    shadow_distance, _ = intersections(points + normals * 0.003 + light * 0.003, light, parts)
    shadow = (np.isfinite(shadow_distance) & valid).astype(np.float32)
    diffuse = np.maximum(np.sum(normals * light, -1), 0) * (1 - shadow)
    halfway = light - direction
    halfway /= np.linalg.norm(halfway)
    specular = np.maximum(np.sum(normals * halfway, -1), 0) ** (12 + 50 * scene.material)
    irradiance = np.clip(0.18 + 0.78 * diffuse + scene.material * 0.25 * specular * (1 - shadow), 0, 1)
    tone = np.where(valid, 1 - irradiance, 0).astype(np.float32)
    # Fade the ground receiver to paper near the image perimeter.
    receiver = np.clip((1.6 - np.sqrt(points[..., 0] ** 2 + points[..., 2] ** 2)) / 0.5, 0, 1)
    # White paper receives only cast-shadow ink; ambient light does not gray it.
    tone[plane] = shadow[plane] * 0.82 * receiver[plane]
    contour = boundaries(index)
    contour *= ((index >= 0) | (boundaries(index >= 0) > 0))
    uy, ux = np.gradient(u)
    length = np.maximum(np.sqrt(ux * ux + uy * uy), 1e-6)
    # A hatch stroke follows a level set of the U coordinate.
    tx, ty = -uy / length, ux / length
    flow = np.stack([tx * tx - ty * ty, 2 * tx * ty], -1)
    depth = np.where(valid, np.clip(distance / 8, 0, 1), 0)
    return {"mask": valid.astype(np.float32), "object": (index >= 0).astype(np.float32),
            "depth": depth.astype(np.float32), "normals": normals.astype(np.float32),
            "contour": contour, "tone": tone, "shadow": shadow, "u": u.astype(np.float32),
            "v": v.astype(np.float32), "flow": flow.astype(np.float32), "light": light,
            "receiver": receiver.astype(np.float32)}
