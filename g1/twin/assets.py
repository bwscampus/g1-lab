"""The twin's two downloads, fetched on demand into ``assets/`` (git-ignored):

  the robot   unitree_mujoco's 29-DoF G1 (torque motors with the real limits, the
              DDS joint order) and its flat scene, BSD-3, Unitree Robotics
  the walker  NVIDIA's GR00T whole-body controller for the G1 (two ONNX networks,
              Balance and Walk), NVIDIA Open Model License

Both come from pinned commits and are checked by sha256. A local clone of
unitree_mujoco next to this repo (or at $G1_UNITREE_MUJOCO) is copied instead
of downloaded when its files match.

    g1 twin fetch
    g1 twin status
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import urllib.request
from pathlib import Path

from g1.core import limits

ASSETS = limits.ROOT / "assets"
MODEL_DIR = ASSETS / "g1_29dof"
SCENE_XML = MODEL_DIR / "scene_29dof.xml"
WALKER_DIR = ASSETS / "walker"
WALKER_BALANCE = WALKER_DIR / "GR00T-WholeBodyControl-Balance.onnx"
WALKER_WALK = WALKER_DIR / "GR00T-WholeBodyControl-Walk.onnx"
WALKER_FILES = (WALKER_BALANCE, WALKER_WALK)

UNITREE_MUJOCO = ("unitreerobotics/unitree_mujoco", "ae6a8403e272733e9996ef59990880330496177f")
GROOT = ("NVlabs/GR00T-WholeBodyControl", "45c03facbf848e0c743091b7d9c17952096ff908")
GROOT_DIR = "decoupled_wbc/sim2mujoco/resources/robots/g1/policy"
SHA256 = {
    "g1_29dof.xml": "423e28bd718b19f7a65cda539b6f794ddbb268b4b9bdbd85f4bd982b30729617",
    "GR00T-WholeBodyControl-Balance.onnx": "f645da599d4ca3d29ed273c8f4712620bb680d34977469ca3aeabe5bb9631c18",
    "GR00T-WholeBodyControl-Walk.onnx": "7c82255b6905ffcc4468fa7f8ddcf7b70db168cf1042107ccab887cb6a8e5407",
}


def _raw(repo: tuple[str, str], path: str) -> str:
    return f"https://raw.githubusercontent.com/{repo[0]}/{repo[1]}/{path}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  {url}")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(dest)


def _local_clone() -> Path | None:
    """A unitree_mujoco checkout whose G1 model is the pinned one."""
    for root in (os.environ.get("G1_UNITREE_MUJOCO"), limits.ROOT.parent / "unitree_mujoco"):
        if not root:
            continue
        xml = Path(root) / "unitree_robots" / "g1" / "g1_29dof.xml"
        if xml.exists() and _sha256(xml) == SHA256["g1_29dof.xml"]:
            return xml.parent
    return None


def meshes(xml: Path) -> list[str]:
    return sorted(set(re.findall(r'<mesh[^>]*\bfile="([^"]+)"', xml.read_text(encoding="utf-8"))))


def missing() -> list[str]:
    out = []
    if not SCENE_XML.exists() or not (MODEL_DIR / "g1_29dof.xml").exists():
        out.append("robot")
    elif any(not (MODEL_DIR / "meshes" / m).exists() for m in meshes(MODEL_DIR / "g1_29dof.xml")):
        out.append("robot meshes")
    if not all(p.exists() for p in WALKER_FILES):
        out.append("walker")
    return out


def require() -> None:
    need = missing()
    if need:
        raise FileNotFoundError(f"twin assets missing ({', '.join(need)}); run: g1 twin fetch")


def fetch(force: bool = False) -> None:
    files = ["g1_29dof.xml", "scene_29dof.xml"]
    if force or "robot" in missing() or "robot meshes" in missing():
        local = _local_clone()
        print(f"robot: unitree_mujoco@{UNITREE_MUJOCO[1][:7]}" + (f" (copied from {local})" if local else ""))
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        for name in files:
            if local:
                shutil.copy2(local / name, MODEL_DIR / name)
            else:
                _download(_raw(UNITREE_MUJOCO, f"unitree_robots/g1/{name}"), MODEL_DIR / name)
        got = _sha256(MODEL_DIR / "g1_29dof.xml")
        if got != SHA256["g1_29dof.xml"]:
            raise RuntimeError(f"g1_29dof.xml sha256 {got} does not match the pinned model")
        for mesh in meshes(MODEL_DIR / "g1_29dof.xml"):
            dest = MODEL_DIR / "meshes" / mesh
            if dest.exists() and not force:
                continue
            if local:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(local / "meshes" / mesh, dest)
            else:
                _download(_raw(UNITREE_MUJOCO, f"unitree_robots/g1/meshes/{mesh}"), dest)
        if local:
            shutil.copy2(local.parents[1] / "LICENSE", MODEL_DIR / "LICENSE")
        else:
            _download(_raw(UNITREE_MUJOCO, "LICENSE"), MODEL_DIR / "LICENSE")
        (MODEL_DIR / "ATTRIBUTION.md").write_text(
            f"G1 29-DoF model from https://github.com/{UNITREE_MUJOCO[0]} at {UNITREE_MUJOCO[1]}\n"
            "(unitree_robots/g1), BSD-3-Clause, Copyright Unitree Robotics. See LICENSE.\n", encoding="utf-8")
    if force or not all(p.exists() for p in WALKER_FILES):
        print(f"walker: GR00T-WholeBodyControl@{GROOT[1][:7]} (NVIDIA Open Model License)")
        for path in WALKER_FILES:              # Git LFS: the media host serves the file, raw only the pointer
            _download(f"https://media.githubusercontent.com/media/{GROOT[0]}/{GROOT[1]}/{GROOT_DIR}/{path.name}", path)
            got = _sha256(path)
            if got != SHA256[path.name]:
                path.unlink()
                raise RuntimeError(f"{path.name} sha256 {got} does not match the pinned walker")
        _download(_raw(GROOT, f"{GROOT_DIR}/NVIDIA%20Open%20Model%20License"), WALKER_DIR / "LICENSE")
        (WALKER_DIR / "ATTRIBUTION.md").write_text(
            f"G1 whole-body controller (Balance, Walk) from https://github.com/{GROOT[0]} at {GROOT[1]}\n"
            f"({GROOT_DIR}), NVIDIA Open Model License, Copyright NVIDIA Corporation. See LICENSE.\n",
            encoding="utf-8")
    print("twin assets: " + ("complete" if not missing() else f"still missing {missing()}"))


def status() -> int:
    need = missing()
    print(f"robot:  {MODEL_DIR}" + ("  MISSING" if "robot" in need or "robot meshes" in need else "  ok"))
    print(f"walker: {WALKER_DIR}" + ("  MISSING" if "walker" in need else "  ok"))
    if need:
        print("run: g1 twin fetch")
    return 1 if need else 0
