from .postprocessor import Postprocessor, PostprocessorCfg
from .postprocessor_none import PostprocessorNone
from .postprocessor_instructir import PostprocessorInstructir

POSTPROCESSORS = {
    "none": PostprocessorNone,
    "instructir": PostprocessorInstructir,
}


def get_postprocessor(postprocessor_cfg: PostprocessorCfg) -> Postprocessor:
    return POSTPROCESSORS[postprocessor_cfg.name](postprocessor_cfg)
