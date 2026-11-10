"""Pose-aware 2D figure studies drawn from joints and tapered contours.

Coordinates are in a 256-pixel drawing space. This is a procedural study target,
not a claim that a few vector shapes can replace curated character artwork.
"""

import numpy as np
from PIL import Image, ImageDraw

FIGURE_VERSION = 3


def render_figure(scene, size):
    rng = np.random.default_rng(scene.geometry_seed)
    armored = scene.shape == "armored_heroine"
    scale = 2
    canvas = 256 * scale
    silhouette = Image.new("L", (canvas, canvas), 0)
    lines = Image.new("L", (canvas, canvas), 0)
    fill = ImageDraw.Draw(silhouette)
    ink = ImageDraw.Draw(lines)
    pose = scene.geometry_seed % 3
    lean = float(rng.uniform(-5, 5))

    def pt(point):
        return tuple(round(float(v) * scale) for v in point)

    def polygon(points, width=1.6):
        points = [pt(p) for p in points]
        fill.polygon(points, fill=255)
        ink.polygon(points, fill=0)
        ink.line(points + points[:1], fill=240, width=max(1, round(width * scale)), joint="curve")

    def stroke(points, width=1.3, shade=220):
        ink.line([pt(p) for p in points], fill=shade, width=max(1, round(width * scale)), joint="curve")

    def curve(a, b, c, d, width=1.3, shade=220):
        stroke(cubic_points(a, b, c, d), width, shade)

    def cubic_points(a, b, c, d):
        t = np.linspace(0, 1, 21)[:, None]
        a, b, c, d = (np.asarray(p, dtype=float) for p in (a, b, c, d))
        return (1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * b + 3 * (1 - t) * t ** 2 * c + t ** 3 * d

    def tapered(a, b, start_width, end_width, bulge=0.0):
        """An asymmetric curved width profile narrows a limb toward its joint."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        axis = b - a
        normal = np.array([-axis[1], axis[0]]) / max(np.linalg.norm(axis), 1)
        t = np.linspace(0, 1, 13)
        smooth = t * t * (3 - 2 * t)
        base = start_width * (1 - smooth) + end_width * smooth
        swell = np.sin(np.pi * t) * bulge * start_width
        upper = [a + s * axis + normal * (w + q) for s, w, q in zip(t, base, swell)]
        lower = [a + s * axis - normal * (0.88 * w + 0.3 * q)
                 for s, w, q in zip(t[::-1], base[::-1], swell[::-1])]
        points = upper + lower
        polygon(points)
        return a, b, normal

    # Three repeatable poses: contrapposto, raised arm, and forward guard.
    cx = 128 + lean
    shoulders = ((cx - 33, 87), (cx + 33, 87))
    hips = ((cx - 19, 152), (cx + 19, 152))
    arm_joints = [
        (((76, 126), (74, 169)), ((180, 129), (190, 169))),
        (((77, 125), (68, 162)), ((181, 67), (168, 32))),
        (((64, 104), (47, 132)), ((190, 103), (211, 125))),
    ][pose]
    leg_joints = [
        (((105, 198), (99, 236)), ((151, 197), (165, 237))),
        (((107, 200), (100, 237)), ((153, 192), (184, 228))),
        (((98, 188), (76, 224)), ((158, 199), (176, 238))),
    ][pose]

    # Legs are articulated thigh/calf forms, not stretched ellipses.
    for hip, (knee, ankle), side in zip(hips, leg_joints, (-1, 1)):
        tapered(hip, knee, 14, 10, 0.10)
        tapered(knee, ankle, 10, 5.5, 0.22)
        foot = np.asarray(ankle, float)
        polygon([foot + (-6, -2), foot + (5, -2), foot + (11 * side, 7),
                 foot + (-9 * side, 7)], 1.5)
        if armored:
            tapered(np.asarray(knee) + (0, 4), np.asarray(ankle) + (0, -5), 9.5, 6.5)
            curve((knee[0] - 8, knee[1] + 4), knee, (knee[0] + 5, knee[1] + 6),
                  (knee[0] + 8, knee[1] + 8), 1.0)
        else:
            stroke([(knee[0] - 5, knee[1]), (knee[0], knee[1] + 3),
                    (knee[0] + 4, knee[1] + 2)], 0.8, 120)

    # Tapered upper arms, narrow wrists, and mitten hands establish clear joints.
    for shoulder, (elbow, wrist), side in zip(shoulders, arm_joints, (-1, 1)):
        tapered(shoulder, elbow, 9.5, 8, 0.06)
        _, _, normal = tapered(elbow, wrist, 8.5, 4.0, 0.22)
        wrist_vec = np.asarray(wrist, float)
        direction = wrist_vec - np.asarray(elbow, float)
        direction /= max(np.linalg.norm(direction), 1)
        polygon([wrist_vec + normal * 4, wrist_vec + direction * 12 + normal * 3,
                 wrist_vec + direction * 16, wrist_vec + direction * 11 - normal * 4,
                 wrist_vec - normal * 4], 1.2)
        stroke([wrist_vec + direction * 7 - normal * 2,
                wrist_vec + direction * 10 + normal * 1], 0.7, 130)
        if armored:
            tapered(np.asarray(elbow) + direction * 4, wrist_vec - direction * 4, 9, 5, 0.10)
            stroke([wrist_vec - direction * 7 - normal * 5,
                    wrist_vec - direction * 7 + normal * 5], 1.4)

    # A narrow waist, broader hips, and curved outer torso produce a figure silhouette.
    left = list(cubic_points((cx - 21, 84), (cx - 39, 91),
                             (cx - 34, 106), (cx - 25, 119)))
    left += list(cubic_points((cx - 25, 119), (cx - 16, 128),
                              (cx - 15, 139), (cx - 25, 152)))[1:]
    right = [(2 * cx - p[0], p[1]) for p in left[::-1]]
    polygon(left + [(cx, 163)] + right + [(cx, 91)])
    if armored:
        polygon([(cx - 27, 98), (cx - 19, 91), (cx, 102), (cx + 19, 91),
                 (cx + 27, 98), (cx + 21, 126), (cx, 138), (cx - 21, 126)], 1.8)
        curve((cx - 19, 112), (cx - 8, 119), (cx + 8, 119), (cx + 19, 112), 1.2)
        stroke([(cx, 102), (cx, 136)], 0.9, 130)
        polygon([(cx - 20, 137), (cx + 20, 137), (cx + 24, 151),
                 (cx, 157), (cx - 24, 151)], 1.5)
        for side in (-1, 1):
            shoulder = np.asarray(shoulders[0 if side < 0 else 1], float)
            polygon([shoulder + (-12 * side, -2), shoulder + (8 * side, -7),
                     shoulder + (18 * side, 6), shoulder + (10 * side, 15),
                     shoulder + (-5 * side, 11)], 1.5)
            stroke([(cx + 19 * side, 144), (cx + 28 * side, 158),
                    (cx + 25 * side, 170)], 1.1)
    else:
        curve((cx - 20, 112), (cx - 9, 117), (cx + 7, 117), (cx + 20, 112), 0.9, 130)
        stroke([(cx - 20, 138), (cx + 20, 138)], 1.2)

    # Hair mass sits behind the face; a pointed jaw and large eyes read as anime.
    hair_cap = list(cubic_points((cx - 25, 58), (cx - 35, 25),
                                  (cx - 11, 12), (cx + 3, 17)))
    hair_cap += list(cubic_points((cx + 3, 17), (cx + 28, 14),
                                   (cx + 31, 45), (cx + 27, 50)))[1:]
    polygon(hair_cap + [(cx + 33, 91), (cx + 17, 82),
                        (cx - 19, 82), (cx - 32, 94)], 1.7)
    polygon([(cx - 20, 39), (cx - 20, 58), (cx - 14, 70), (cx, 78),
             (cx + 14, 70), (cx + 20, 58), (cx + 18, 37), (cx, 29)], 1.7)
    polygon([(cx - 24, 41), (cx - 15, 22), (cx + 3, 20), (cx + 22, 30),
             (cx + 15, 41), (cx + 9, 35), (cx + 1, 46), (cx - 8, 37),
             (cx - 18, 48)], 1.3)
    for side in (-1, 1):
        eye_x = cx + side * 9
        curve((eye_x - 5, 54), (eye_x - 2, 49), (eye_x + 2, 49), (eye_x + 5, 54), 1.5)
        stroke([(eye_x - 3, 55), (eye_x, 58), (eye_x + 3, 55)], 0.9)
        stroke([(eye_x - 4, 47), (eye_x + 3, 46)], 0.8, 140)
    stroke([(cx - 2, 65), (cx, 66), (cx + 2, 65)], 0.7, 150)
    stroke([(cx - 3, 70), (cx, 71), (cx + 3, 70)], 0.8, 170)
    curve((cx - 24, 48), (cx - 29, 65), (cx - 29, 84), (cx - 31, 101), 1.4)
    curve((cx + 24, 48), (cx + 31, 65), (cx + 25, 82), (cx + 32, 98), 1.4)

    resampling = Image.Resampling.LANCZOS
    mask = np.asarray(silhouette.resize((size, size), resampling), dtype=np.float32) / 255
    line_art = np.asarray(lines.resize((size, size), resampling), dtype=np.float32) / 255
    axis = np.linspace(-1, 1, size, dtype=np.float32)
    x, y = np.meshgrid(axis, axis)
    light_xy = np.array([np.sin(np.deg2rad(scene.light_azimuth)),
                         -np.cos(np.deg2rad(scene.light_elevation))], dtype=np.float32)
    light_xy /= max(np.linalg.norm(light_xy), 1e-6)
    shade = np.clip(0.14 + 0.12 * (x * -light_xy[0] + y * -light_xy[1]), 0.03, 0.28)
    tone = (mask * shade).astype(np.float32)
    normals = np.zeros((size, size, 3), dtype=np.float32)
    normals[..., 2] = 1
    # The line map is an explicit structural guide. The model renders its ink;
    # it does not invent facial features or armor construction from a silhouette.
    contour = line_art.astype(np.float32)
    flow = np.zeros((size, size, 2), dtype=np.float32)
    flow[..., 0] = 1
    return {"mask": mask, "object": mask, "depth": mask * 0.5, "normals": normals,
            "contour": contour, "tone": tone, "shadow": np.zeros_like(mask),
            "u": x, "v": y, "flow": flow, "light": np.array([*light_xy, 0], dtype=np.float32),
            "receiver": np.zeros_like(mask), "line_art": line_art}
