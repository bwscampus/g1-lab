"""A real-looking room for the sim, so the real vision model has something
real to look at: floor and walls with photo textures, a table and chairs from
primitives, and real textured meshes for the objects a search is about.

Assets are fetched on demand into ``assets/`` (git-ignored) with attribution:
  Poly Haven textures (CC0)                 https://polyhaven.com
  YCB object meshes (CC-BY 4.0, Calli et al.)  https://www.ycbbenchmarks.com

    python -m scene fetch            # everything (about 40 MB)
    python -m scene fetch mug pencil # just what a run needs (pencil is built from primitives)
    python -m scene status

Robot frame: it starts at the origin facing +x. The room spans x -2..4,
y -3..3, 2.5 m high, with a doorway in the +x wall.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Optional, Sequence

ASSETS = Path(__file__).parent / "assets"

TEXTURES = {   # name: (Poly Haven slug, jpg url); stored as PNG (MuJoCo does not read JPEG)
    "floor": ("laminate_floor_02",
              "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/laminate_floor_02/laminate_floor_02_diff_1k.jpg"),
    "wall": ("painted_plaster_wall",
             "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/painted_plaster_wall/painted_plaster_wall_diff_1k.jpg"),
}
YCB = {        # object name: YCB id (google_16k scans: textured.obj + texture_map.png, metres)
    "mug": "025_mug",
    "marker": "040_large_marker",
    "cracker_box": "003_cracker_box",
    "mustard": "006_mustard_bottle",
}
YCB_URL = "https://ycb-benchmarks.s3.amazonaws.com/data/google/{id}_google_16k.tgz"
PRIMITIVE_OBJECTS = ("pencil",)
OBJECTS = tuple(YCB) + PRIMITIVE_OBJECTS

# Room layout (metres)
ROOM_X = (-2.0, 4.0)
ROOM_Y = (-3.0, 3.0)
ROOM_H = 2.5
DOOR_Y = (0.5, 1.4)          # gap in the +x wall
DOOR_H = 2.0
TABLE = (2.2, -1.5)           # centre; 1.2 x 0.7 m top at 0.75 m


# --------------------------------------------------------------------------
# Assets
# --------------------------------------------------------------------------

def texture_path(name: str) -> Path:
    return ASSETS / "textures" / f"{TEXTURES[name][0]}.png"


def ycb_dir(name: str) -> Path:
    return ASSETS / "ycb" / YCB[name]


def ycb_files(name: str) -> tuple[Path, Path]:
    d = ycb_dir(name)
    return d / "google_16k" / "textured.obj", d / "google_16k" / "texture_map.png"


def missing(objects: Sequence[str] = ()) -> list[str]:
    """Asset names that a room with these objects still needs."""
    out = [f"texture:{n}" for n in TEXTURES if not texture_path(n).exists()]
    for o in objects:
        if o in YCB and not all(p.exists() for p in ycb_files(o)):
            out.append(f"ycb:{o}")
    return out


def _download(url: str, dest: Path, label: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching {label} ... ", end="", flush=True)
    with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
        n = 0
        while chunk := r.read(1 << 16):
            f.write(chunk)
            n += len(chunk)
    print(f"{n / 1e6:.1f} MB")


def fetch(objects: Sequence[str] = tuple(YCB), force: bool = False) -> None:
    import cv2
    for name, (slug, url) in TEXTURES.items():
        png = texture_path(name)
        if png.exists() and not force:
            continue
        jpg = png.with_suffix(".jpg")
        _download(url, jpg, f"texture {slug}")
        img = cv2.imread(str(jpg))
        if img is None or not cv2.imwrite(str(png), img):
            raise IOError(f"could not convert {jpg}")
        jpg.unlink()
    for o in objects:
        if o not in YCB:
            continue
        obj, png = ycb_files(o)
        if obj.exists() and png.exists() and not force:
            continue
        tgz = ASSETS / "ycb" / f"{YCB[o]}.tgz"
        _download(YCB_URL.format(id=YCB[o]), tgz, f"YCB {YCB[o]}")
        with tarfile.open(tgz) as t:
            members = [m for m in t.getmembers() if m.name.endswith(("textured.obj", "textured.mtl", "texture_map.png"))]
            t.extractall(ASSETS / "ycb", members=members)
        tgz.unlink()
    (ASSETS / "ATTRIBUTION.md").write_text(
        "# Third-party assets (downloaded by `python -m scene fetch`, not part of this repo)\n\n"
        "## Textures — Poly Haven, CC0 (https://polyhaven.com/license)\n"
        + "".join(f"- {slug}: {url}\n" for slug, url in TEXTURES.values())
        + "\n## Object meshes — YCB Object and Model Set, CC-BY 4.0\n"
        "Calli, Singh, Walsman, Srinivasa, Abbeel, Dollar: *The YCB Object and Model Set*, ICAR 2015.\n"
        "https://www.ycbbenchmarks.com — google_16k scans:\n"
        + "".join(f"- {n}: {YCB_URL.format(id=i)}\n" for n, i in YCB.items()))


def _obj_bounds(path: Path):
    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                x, y, z = (float(v) for v in line.split()[1:4])
                for i, v in enumerate((x, y, z)):
                    lo[i] = min(lo[i], v)
                    hi[i] = max(hi[i], v)
    return lo, hi


# --------------------------------------------------------------------------
# Building
# --------------------------------------------------------------------------

def parse_object(spec: str) -> tuple[str, float, float, Optional[float]]:
    """``name@x,y[,z]``; z defaults to resting on the floor."""
    m = re.fullmatch(r"\s*([a-z_]+)@(-?[\d.]+),(-?[\d.]+)(?:,(-?[\d.]+))?\s*", spec)
    if not m or m.group(1) not in OBJECTS:
        raise argparse.ArgumentTypeError(f"expected name@x,y[,z] with name in {OBJECTS}, got {spec!r}")
    return m.group(1), float(m.group(2)), float(m.group(3)), None if m.group(4) is None else float(m.group(4))


def build_room(spec, objects: Sequence[tuple[str, float, float, Optional[float]]] = (),
               camera_size: Optional[tuple[int, int]] = None) -> None:
    """Add the room, furniture, lights and objects to an MjSpec of the G1 scene."""
    import mujoco

    need = missing([o[0] for o in objects])
    if need:
        raise FileNotFoundError(f"room assets missing ({', '.join(need)}); run: python -m scene fetch")
    if camera_size is not None:
        spec.visual.global_.offheight = max(spec.visual.global_.offheight, camera_size[0])
        spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, camera_size[1])

    def material(name: str, tex: str, repeat: float, rgba=(1, 1, 1, 1)) -> str:
        spec.add_texture(name=f"{name}_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(texture_path(tex)))
        mat = spec.add_material(name=f"{name}_mat", texrepeat=[repeat, repeat], rgba=list(rgba))
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"{name}_tex"
        return f"{name}_mat"

    floor = material("room_floor", "floor", 8)
    wall = material("room_wall", "wall", 3)
    # Furniture: the floor laminate tinted to a darker wood. Kept deliberately un-red so the
    # pixel red-blob path (RedDot, the test double) is not fooled by orange wood.
    wood = material("room_wood", "floor", 2, rgba=(0.55, 0.48, 0.42, 1))
    B = mujoco.mjtGeom.mjGEOM_BOX
    w = spec.worldbody
    cx, cy = (ROOM_X[0] + ROOM_X[1]) / 2, (ROOM_Y[0] + ROOM_Y[1]) / 2
    hx, hy = (ROOM_X[1] - ROOM_X[0]) / 2, (ROOM_Y[1] - ROOM_Y[0]) / 2
    w.add_geom(name="room_floor", type=B, size=[hx, hy, 0.005], pos=[cx, cy, 0.005], material=floor,
               contype=0, conaffinity=0)
    t, hz = 0.05, ROOM_H / 2
    w.add_geom(name="wall_-x", type=B, size=[t, hy, hz], pos=[ROOM_X[0], cy, hz], material=wall)
    w.add_geom(name="wall_-y", type=B, size=[hx, t, hz], pos=[cx, ROOM_Y[0], hz], material=wall)
    w.add_geom(name="wall_+y", type=B, size=[hx, t, hz], pos=[cx, ROOM_Y[1], hz], material=wall)
    # +x wall with a doorway
    y0, y1 = DOOR_Y
    w.add_geom(name="wall_+x_a", type=B, size=[t, (y0 - ROOM_Y[0]) / 2, hz], pos=[ROOM_X[1], (ROOM_Y[0] + y0) / 2, hz], material=wall)
    w.add_geom(name="wall_+x_b", type=B, size=[t, (ROOM_Y[1] - y1) / 2, hz], pos=[ROOM_X[1], (y1 + ROOM_Y[1]) / 2, hz], material=wall)
    w.add_geom(name="wall_+x_lintel", type=B, size=[t, (y1 - y0) / 2, (ROOM_H - DOOR_H) / 2],
               pos=[ROOM_X[1], (y0 + y1) / 2, (ROOM_H + DOOR_H) / 2], material=wall)
    # table and two chairs
    tx, ty = TABLE
    w.add_geom(name="table_top", type=B, size=[0.6, 0.35, 0.02], pos=[tx, ty, 0.75], material=wood)
    for i, (dx, dy) in enumerate(((0.55, 0.3), (-0.55, 0.3), (0.55, -0.3), (-0.55, -0.3))):
        w.add_geom(name=f"table_leg{i}", type=B, size=[0.03, 0.03, 0.365], pos=[tx + dx, ty + dy, 0.365], material=wood)
    for k, (chx, chy, back_dy) in enumerate(((tx - 0.3, ty + 0.75, 0.2), (tx + 0.3, ty + 0.75, 0.2))):
        w.add_geom(name=f"chair{k}_seat", type=B, size=[0.22, 0.22, 0.02], pos=[chx, chy, 0.45], material=wood)
        w.add_geom(name=f"chair{k}_back", type=B, size=[0.22, 0.02, 0.25], pos=[chx, chy + back_dy, 0.72], material=wood)
        for i, (dx, dy) in enumerate(((0.18, 0.18), (-0.18, 0.18), (0.18, -0.18), (-0.18, -0.18))):
            w.add_geom(name=f"chair{k}_leg{i}", type=B, size=[0.02, 0.02, 0.215], pos=[chx + dx, chy + dy, 0.215], material=wood)
    w.add_light(name="room_light_a", pos=[0.5, 0.0, 2.4], dir=[0, 0, -1], diffuse=[0.7, 0.7, 0.7])
    w.add_light(name="room_light_b", pos=[2.5, 1.5, 2.4], dir=[0, 0, -1], diffuse=[0.5, 0.5, 0.5])
    # objects
    counts: dict[str, int] = {}
    for name, x, y, z in objects:
        counts[name] = counts.get(name, 0) + 1
        tag = f"{name}{counts[name]}"
        if name in YCB:
            obj, png = ycb_files(name)
            lo, _ = _obj_bounds(obj)
            spec.add_mesh(name=f"{tag}_mesh", file=str(obj))
            spec.add_texture(name=f"{tag}_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(png))
            mat = spec.add_material(name=f"{tag}_mat")
            mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"{tag}_tex"
            body = w.add_body(name=tag, pos=[x, y, (-lo[2] if z is None else z)])
            body.add_geom(name=f"{tag}_geom", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"{tag}_mesh", material=f"{tag}_mat")
        elif name == "pencil":
            body = w.add_body(name=tag, pos=[x, y, (0.0035 if z is None else z)])
            lying = [0.7071068, 0.0, 0.7071068, 0.0]        # cylinder axis z -> x
            C = mujoco.mjtGeom.mjGEOM_CYLINDER
            body.add_geom(name=f"{tag}_body", type=C, size=[0.0035, 0.085], quat=lying, rgba=[0.95, 0.75, 0.1, 1])
            body.add_geom(name=f"{tag}_eraser", type=C, size=[0.0037, 0.008], pos=[-0.093, 0, 0], quat=lying, rgba=[0.95, 0.55, 0.6, 1])
            body.add_geom(name=f"{tag}_ferrule", type=C, size=[0.0037, 0.004], pos=[-0.081, 0, 0], quat=lying, rgba=[0.75, 0.75, 0.78, 1])
            body.add_geom(name=f"{tag}_tip", type=C, size=[0.002, 0.008], pos=[0.093, 0, 0], quat=lying, rgba=[0.85, 0.7, 0.5, 1])


def _main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m scene", description="fetch or inspect the room's assets")
    p.add_argument("command", choices=("fetch", "status"))
    p.add_argument("objects", nargs="*", help=f"object names to fetch (default: all of {list(YCB)})")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    if args.command == "fetch":
        bad = [o for o in args.objects if o not in OBJECTS]
        if bad:
            p.error(f"unknown objects {bad}; choose from {OBJECTS}")
        fetch(args.objects or tuple(YCB), force=args.force)
        print(f"assets in {ASSETS}; attribution in {ASSETS / 'ATTRIBUTION.md'}")
    need = missing(tuple(YCB))
    print("missing: " + (", ".join(need) if need else "nothing") + f"  (objects: {', '.join(OBJECTS)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
