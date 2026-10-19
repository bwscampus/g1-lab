"""The twin's two downloads, fetched on demand into ``assets/`` (git-ignored):

  the robot   unitree_mujoco's 29-DoF G1 (torque motors with the real limits, the
              DDS joint order) and its flat scene, BSD-3, Unitree Robotics
  the walker  unitree_rl_gym's trained G1 gait network (TorchScript), BSD-3,
              Unitree Robotics

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
WALKER = WALKER_DIR / "motion.pt"

UNITREE_MUJOCO = ("unitreerobotics/unitree_mujoco", "ae6a8403e272733e9996ef59990880330496177f")
RL_GYM = ("unitreerobotics/unitree_rl_gym", "276801e46c5d433564f24658bac64f254b7d2d4b")
SHA256 = {
    "g1_29dof.xml": "423e28bd718b19f7a65cda539b6f794ddbb268b4b9bdbd85f4bd982b30729617",
    "motion.pt": "cf668f75b90d1abf73d2b87612a6e76bccc61ff7e083b63582d3f6aaa3c1759d",
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
    if not WALKER.exists():
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
    if force or not WALKER.exists():
        print(f"walker: unitree_rl_gym@{RL_GYM[1][:7]}")
        _download(_raw(RL_GYM, "deploy/pre_train/g1/motion.pt"), WALKER)
        got = _sha256(WALKER)
        if got != SHA256["motion.pt"]:
            WALKER.unlink()
            raise RuntimeError(f"motion.pt sha256 {got} does not match the pinned walker")
        _download(_raw(RL_GYM, "LICENSE"), WALKER_DIR / "LICENSE")
        (WALKER_DIR / "ATTRIBUTION.md").write_text(
            f"G1 gait network from https://github.com/{RL_GYM[0]} at {RL_GYM[1]}\n"
            "(deploy/pre_train/g1/motion.pt), BSD-3-Clause, Copyright Unitree Robotics. See LICENSE.\n",
            encoding="utf-8")
    print("twin assets: " + ("complete" if not missing() else f"still missing {missing()}"))


def status() -> int:
    need = missing()
    print(f"robot:  {MODEL_DIR}" + ("  MISSING" if "robot" in need or "robot meshes" in need else "  ok"))
    print(f"walker: {WALKER}" + ("  MISSING" if "walker" in need else "  ok"))
    if need:
        print("run: g1 twin fetch")
    return 1 if need else 0
