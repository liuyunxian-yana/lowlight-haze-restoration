"""独立项目的数据目录、严格配对与输入校验。

恢复算法只接收 verify 返回的输入 RGB；参考图和清单参数仅供评估。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .core import validate

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})


@dataclass(frozen=True)
class Case:
    id: str
    source_id: str
    split: str
    kind: str
    severity: str
    input_path: Path
    reference_path: Path | None
    input_sha256: str | None = None
    reference_sha256: str | None = None


def _inside(root: Path, relative: str) -> Path:
    """清单里的路径只能位于自己的数据目录。"""
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"数据清单路径越界：{relative}")
    return path


def _index(folder: Path, suffix: str = "") -> dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"图像目录不存在：{folder}")
    found = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        if suffix and not path.stem.endswith(suffix):
            continue
        key = path.stem[:-len(suffix)] if suffix else path.stem
        if not key or key in found:
            raise ValueError(f"图像配对名称为空或重复：{path.name}")
        found[key] = path.resolve()
    if not found:
        raise ValueError(f"未找到图像：{folder}")
    return found


def _paired(inputs: Path, references: Path | None, input_suffix: str = "", reference_suffix: str = "", kind: str = "external") -> list[Case]:
    image_index = _index(inputs, input_suffix)
    reference_index = _index(references, reference_suffix) if references else None
    if reference_index is not None and image_index.keys() != reference_index.keys():
        missing_ref = sorted(image_index.keys() - reference_index.keys())[:5]
        missing_input = sorted(reference_index.keys() - image_index.keys())[:5]
        raise ValueError(f"输入与参考图未严格配对；缺参考={missing_ref}，缺输入={missing_input}")
    return [Case(id=key, source_id=key, split="external", kind=kind,
                   severity="unspecified", input_path=image_index[key],
                   reference_path=reference_index[key] if reference_index else None)
            for key in sorted(image_index)]


def _synthetic(split: str, kind: str | None, data_root: Path | None = None) -> list[Case]:
    if split not in {"val", "test"}:
        raise ValueError("合成数据 split 只能是 val 或 test")
    if kind not in {None, "all", "lowlight", "haze", "joint"}:
        raise ValueError("kind 只能是 lowlight、haze、joint 或 all")
    root = Path(data_root) if data_root is not None else DATA / "synthetic"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, list):
        raise ValueError("manifest.json 必须是样本列表")
    id_set, split_by_source, split_by_reference = set(), {}, {}
    reference_by_source, source_by_reference = {}, {}
    selected = []
    for row in manifest:
        sample_id, source_id, row_split = row["id"], row["source_id"], row["split"]
        if sample_id in id_set:
            raise ValueError(f"manifest 样本 ID 重复：{sample_id}")
        id_set.add(sample_id)
        ref_identity = row.get("reference_sha256") or row["reference"]
        for key, mapping in ((source_id, split_by_source), (ref_identity, split_by_reference)):
            if key in mapping and mapping[key] != row_split:
                raise ValueError("同一原图/参考内容跨验证和测试集，存在数据泄漏")
            mapping[key] = row_split
        if source_id in reference_by_source and reference_by_source[source_id] != ref_identity:
            raise ValueError("同一原图 ID 对应多个参考内容")
        if ref_identity in source_by_reference and source_by_reference[ref_identity] != source_id:
            raise ValueError("不同原图 ID 共用同一参考内容")
        reference_by_source[source_id] = ref_identity
        source_by_reference[ref_identity] = source_id
        if row_split == split and (kind in (None, "all") or row["kind"] == kind):
            selected.append(Case(id=sample_id, source_id=source_id, split=row_split,
                                   kind=row["kind"], severity=row["severity"],
                                   input_path=_inside(root, row["input"]),
                                   reference_path=_inside(root, row["reference"]),
                                   input_sha256=row.get("input_sha256"),
                                   reference_sha256=row.get("reference_sha256")))
    if not selected:
        raise ValueError(f"未找到合成数据样本：split={split}, kind={kind}")
    return selected


def load(dataset: str = "synthetic", split: str = "test", kind: str | None = "joint",
                 input_dir: Path | None = None, reference_dir: Path | None = None,
                 input_suffix: str = "", reference_suffix: str = "", data_root: Path | None = None) -> list[Case]:
    """返回完整配对表；不将参考图、噪声参数或合成图传给算法。"""
    if dataset == "synthetic":
        return _synthetic(split, kind, data_root)
    if dataset == "nthazy":
        root = DATA / "nthazy"
        return _paired(root / "input", root / "gt", "_NTHazy", "_GT", "haze")
    if dataset == "gta5":
        root = DATA / "gta5"
        return _paired(root / "foggy", root / "clean", kind="haze")
    if dataset == "external":
        if input_dir is None:
            raise ValueError("external 数据集必须设置 input_dir")
        return _paired(Path(input_dir), Path(reference_dir) if reference_dir else None,
                       input_suffix, reference_suffix)
    raise ValueError(f"未知数据集：{dataset}")


def read_rgb(path: Path) -> np.ndarray:
    """路径按字节解码以支持中文文件名；只对输入/参考作统一 RGB 转换。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    bgr = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"无法解码图像：{path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    validate(rgb)
    return rgb


def write_rgb(path: Path, image: np.ndarray) -> None:
    validate(image)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix not in IMAGE_EXTS:
        raise ValueError(f"不支持的图像格式：{suffix}")
    ok, encoded = cv2.imencode(suffix, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise OSError(f"图像编码失败：{path}")
    path.write_bytes(encoded.tobytes())


def verify(sample: Case, check_hash: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
    """验证存在、哈希及尺寸，并读入一对图像。无参考图返回 None。"""
    for path, expected in ((sample.input_path, sample.input_sha256),
                           (sample.reference_path, sample.reference_sha256)):
        if path is None:
            continue
        if not path.is_file():
            raise FileNotFoundError(path)
        if check_hash and expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"数据 SHA-256 不匹配：{path}")
    image = read_rgb(sample.input_path)
    reference = read_rgb(sample.reference_path) if sample.reference_path is not None else None
    if reference is not None and image.shape != reference.shape:
        raise ValueError(f"输入与参考尺寸不一致，不自动缩放：{image.shape} vs {reference.shape}")
    return image, reference
