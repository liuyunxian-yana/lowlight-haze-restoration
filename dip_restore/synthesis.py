"""离线生成新实验数据；物理顺序与课程数据一致，插值实现版本单独标记。"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from .core import linear_to_srgb, srgb_to_linear
from .datasets import read_rgb, write_rgb

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_for(base, source, severity):
    return int.from_bytes(hashlib.sha256(f"{base}:{source}:{severity}".encode()).digest()[:8], "little")


def degrade(clean, params, seed):
    """线性域：先雾，再空间曝光，最后 Poisson+Gaussian 传感器噪声。"""
    h, w = clean.shape[:2]
    rng = np.random.default_rng(seed)
    field = cv2.resize(rng.random((8, 8)).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    illumination = cv2.resize(rng.random((6, 6)).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    vertical = np.linspace(1, 0, h, dtype=np.float32)[:, None]
    depth = 0.15 + 0.85 * (0.55 * vertical + 0.45 * field)
    t = np.exp(-float(params["beta"]) * depth)[..., None]
    e = float(params["exposure"]) * (0.7 + 0.3 * illumination)[..., None]
    a = np.asarray(params["airlight_linear"], np.float32)
    scene = srgb_to_linear(clean.astype(np.float32) / 255)
    hazy = scene * t + a * (1 - t)

    def capture(signal, noise_seed):
        local = np.random.default_rng(noise_seed)
        counts = float(params["photon_count"])
        shot = local.poisson(np.clip(signal, 0, 1) * counts) / counts
        read = local.normal(0, float(params["read_sigma"]), signal.shape)
        return np.clip(shot + read, 0, 1)

    low = capture(e * scene, seed + 1)
    joint = capture(e * hazy, seed + 1)
    return {"lowlight": np.rint(linear_to_srgb(low) * 255).astype(np.uint8),
            "haze": np.rint(linear_to_srgb(hazy) * 255).astype(np.uint8),
            "joint": np.rint(linear_to_srgb(joint) * 255).astype(np.uint8)}


def generate(clean_root, out, config_path=None, limit_sources=None):
    clean_root, out = Path(clean_root), Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("生成目录非空")
    config_path = Path(config_path) if config_path else ROOT / "data/synthetic/synthesis_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    sources = [(split, path) for split in ("val", "test")
               for path in sorted((clean_root / split).glob("*.jpg"))]
    if limit_sources:
        sources = sources[:limit_sources]
    if not sources:
        raise ValueError("找不到 val/test 下的 JPG 清晰图")
    rows, provenance = [], []
    for split, source in sources:
        rel_clean = Path("clean") / split / source.name
        target = out / rel_clean
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        clean = read_rgb(target)
        source_id = source.stem
        provenance.append(dict(source_id=source_id, split=split, clean=str(rel_clean), sha256=sha(target)))
        for severity, params in config["levels"].items():
            seed = seed_for(config["seed"], source_id, severity)
            for kind, image in degrade(clean, params, seed).items():
                sample_id = f"{source_id}_{kind}_{severity}"
                rel_input = Path("degraded") / split / f"{sample_id}.png"
                write_rgb(out / rel_input, image)
                rows.append(dict(id=sample_id, source_id=source_id, split=split, kind=kind,
                                 severity=severity, input=str(rel_input), reference=str(rel_clean),
                                 input_sha256=sha(out / rel_input), reference_sha256=sha(target),
                                 seed=seed, params=params, model="unified_cv2_linear_haze_exposure_v1"))
    out.mkdir(parents=True, exist_ok=True)
    for name, value in (("manifest.json", rows), ("sources.json", provenance), ("synthesis_config.json", config)):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, default=ROOT / "data/synthetic/clean")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit-sources", type=int)
    args = parser.parse_args()
    print(f"生成 {generate(args.clean, args.out, limit_sources=args.limit_sources)} 张图；这是新生成器版本，不能与内置原数据逐字节比较")


if __name__ == "__main__":
    main()
