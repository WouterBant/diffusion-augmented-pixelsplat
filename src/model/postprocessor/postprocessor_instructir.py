from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from .postprocessor import Postprocessor, PostprocessorCfg
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import yaml
import random

from instructir.utils import *
from instructir.models import instructir
from huggingface_hub import hf_hub_download

from instructir.text.models import LanguageModel, LMHead

#SEED=42
#seed_everything(SEED=SEED)
#torch.backends.cudnn.deterministic = True

use_inpainting = False

CONFIG = "instructir/configs/eval5d.yml"
USE_WANDB = False
LM_MODEL = "instructir/models/lm_instructir-7d.pt"
MODEL_NAME = "instructir/models/im_instructir-7d.pt"  # original model
if use_inpainting:
    MODEL_NAME = hf_hub_download(repo_id="Wouter01/InstructIR_with_inpainting", filename="alternative.pt")
else:
    MODEL_NAME = hf_hub_download(repo_id="Wouter01/instructir_hard_data", filename="instructir_hard_data/best_model.pt")

# parse config file
with open(os.path.join(CONFIG), "r") as f:
    config = yaml.safe_load(f)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

external_cfg = dict2namespace(config)


class PostprocessorInstructir(Postprocessor[PostprocessorCfg]):
    def __init__(
        self,
        cfg: PostprocessorCfg,
    ) -> None:
        super().__init__(cfg)

        print("Creating InstructIR")
        self.model = instructir.create_model(
            input_channels=external_cfg.model.in_ch,
            width=external_cfg.model.width,
            enc_blks=external_cfg.model.enc_blks,
            middle_blk_num=external_cfg.model.middle_blk_num,
            dec_blks=external_cfg.model.dec_blks,
            txtdim=external_cfg.model.textdim)

        ################### LOAD IMAGE MODEL

        assert MODEL_NAME, "Model weights required for evaluation"

        print("IMAGE MODEL CKPT:", MODEL_NAME)
        self.model.load_state_dict(torch.load(MODEL_NAME,
                                              # map_location=device
                                              ), strict=True)

        nparams = count_params(self.model)
        print("Loaded weights!", nparams / 1e6)

        ################### LANGUAGE MODEL
        if external_cfg.model.use_text:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

            # Initialize the LanguageModel class
            self.LMODEL = external_cfg.llm.model
            self.language_model = LanguageModel(model=self.LMODEL)
            self.lm_head = LMHead(embedding_dim=external_cfg.llm.model_dim,
                             hidden_dim=external_cfg.llm.embd_dim,
                             num_classes=external_cfg.llm.nclasses)
            self.lm_head = self.lm_head  # .to(device)
            self.lm_nparams = count_params(self.lm_head)

            print("LMHEAD MODEL CKPT:", self.LM_MODEL)
            self.lm_head.load_state_dict(torch.load(LM_MODEL,
                                                    # map_location=device
                                                    ), strict=True)
            print("Loaded weights!")

        else:
            self.LMODEL = None
            self.language_model = None
            self.lm_head = None
            self.lm_nparams = 0

    def forward(
            self,
            img: Float[Tensor, "batch view 3 height width"]
    ) -> Float[Tensor, "batch view 3 height width"]:
        # Convert the image to tensor

        # y = torch.Tensor(image).permute(2, 0, 1).unsqueeze(0)
        y = img.unsqueeze(0)

        # Get the text embedding (and predicted degradation class)
        prompt = ""  # FIXME
        lm_embd = self.language_model(prompt)
        lm_embd = lm_embd  # .to(device)
        text_embd, deg_pred = self.lm_head(lm_embd)

        # Forward pass: Paper Figure 2
        iterations = 1
        for _ in range(iterations):
            y = self.model(y, text_embd)


        return y
        # convert the restored image <x_hat> into a np array
        #restored_img = y[0].permute(1, 2, 0).cpu().detach().numpy()
        #restored_img = np.clip(restored_img, 0., 1.)
        # return restored_img
