from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from ...dataset import DatasetCfg
from ..types import Gaussians
from .cuda_splatting import DepthRenderingMode, render_cuda, render_depth_cuda
from .postprocessor import Postprocessor, PostprocessorCfg


class PostprocessorNone(Postprocessor[PostprocessorCfg]):
    def __init__(
        self,
        cfg: PostprocessorCfg,
    ) -> None:
        super().__init__(cfg)

    def forward(
            self,
            img: Float[Tensor, "batch view 3 height width"]
    ) -> Float[Tensor, "batch view 3 height width"]:
        return img
