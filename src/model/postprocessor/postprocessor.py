from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from jaxtyping import Float
from torch import Tensor, nn


# @dataclass
# class DecoderOutput:
#     color: Float[Tensor, "batch view 3 height width"]
#     depth: Float[Tensor, "batch view height width"] | None
#

T = TypeVar("T")


@dataclass
class PostprocessorCfg:
    name: Literal["none", "instructir", "controlnet"]
    # model: Literal["instructir", "controlnet"]


class Postprocessor(nn.Module, ABC, Generic[T]):
    cfg: T
    # dataset_cfg: DatasetCfg

    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg
        # self.dataset_cfg = dataset_cfg

    @abstractmethod
    def forward(
        self,
        img: Float[Tensor, "batch view 3 height width"]
    ) -> Float[Tensor, "batch view 3 height width"]:
        pass
