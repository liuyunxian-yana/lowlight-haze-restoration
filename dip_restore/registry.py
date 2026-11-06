"""所有算法通过同一个 RGB uint8 函数注册；完整联合方法不误当单步骤。"""
from __future__ import annotations

from .core import validate
from .dehazing import DEHAZERS
from .enhancement import ENHANCERS
from .joint_model import JOINT_MODEL
from .prior_joint import PRIOR_JOINT
from .rdt import RDT_METHODS

SINGLE = {**ENHANCERS, **DEHAZERS}
JOINT = {**JOINT_MODEL, **PRIOR_JOINT, **RDT_METHODS}
ALL = {"identity": lambda x: x.copy(), **SINGLE, **JOINT}


def parse(spec):
    if spec in ALL:
        return (spec,)
    parts = spec.split(">")
    if len(parts) == 2 and ((parts[0] in ENHANCERS and parts[1] in DEHAZERS) or
                            (parts[0] in DEHAZERS and parts[1] in ENHANCERS)):
        return tuple(parts)
    raise ValueError(f"无效算法规格 {spec!r}；可用单项、联合方法，或 gamma>dcp / dcp>gamma 形式")


def restore(image, spec):
    validate(image)
    output = image.copy()
    for name in parse(spec):
        output = ALL[name](output)
        validate(output, image.shape)
    return output


def filename(spec):
    return spec.replace(">", "__then__")
