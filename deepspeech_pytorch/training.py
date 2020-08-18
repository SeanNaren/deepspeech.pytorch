import json
import random

import numpy as np
import torch.utils.data.distributed
import pytorch_lightning as pl
from hydra.utils import to_absolute_path

from deepspeech_pytorch.configs.train_config import DeepSpeechConfig
from deepspeech_pytorch.loader.data_module import DeepSpeechDataModule
from deepspeech_pytorch.logger import TrainsLogger
from deepspeech_pytorch.model import DeepSpeech


def train(cfg: DeepSpeechConfig):
    # Set seeds for determinism
    torch.manual_seed(cfg.training.seed)
    torch.cuda.manual_seed_all(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    random.seed(cfg.training.seed)

    with open(to_absolute_path(cfg.data.labels_path)) as label_file:
        labels = json.load(label_file)

    # dataloader
    data_loader = DeepSpeechDataModule(
        labels=labels,
        data_cfg=cfg.data,
        epoch=0,
        training_step=0,
        normalize=True
    )

    # init model
    model = DeepSpeech(
        labels=labels,
        model_cfg=cfg.model,
        optim_cfg=cfg.optim,
        spect_cfg=cfg.data.spect
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        gpus=cfg.training.gpus,
        logger=[TrainsLogger()]
    )
    trainer.fit(model, data_loader)
