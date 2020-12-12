import json

import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer

from deepspeech_pytorch.checkpoint import CheckpointHandler, GCSCheckpointHandler
from deepspeech_pytorch.configs.train_config import DeepSpeechConfig, GCSCheckpointConfig
from deepspeech_pytorch.loader.data_module import DeepSpeechDataModule
from deepspeech_pytorch.model import DeepSpeech


def train(cfg: DeepSpeechConfig):
    # Set seeds for determinism
    seed_everything(cfg.seed)

    with open(to_absolute_path(cfg.data.labels_path)) as label_file:
        labels = json.load(label_file)

    if cfg.trainer.checkpoint_callback:
        if OmegaConf.get_type(cfg.checkpoint) is GCSCheckpointConfig:
            checkpoint_callback = GCSCheckpointHandler(
                cfg=cfg.checkpoint
            )
            cfg.trainer.callbacks = [checkpoint_callback]
        else:
            checkpoint_callback = CheckpointHandler(
                cfg=cfg.checkpoint
            )
        if cfg.load_auto_checkpoint:
            resume_from_checkpoint = checkpoint_callback.find_latest_checkpoint()
            if resume_from_checkpoint:
                cfg.trainer.resume_from_checkpoint = resume_from_checkpoint

    data_loader = DeepSpeechDataModule(
        labels=labels,
        data_cfg=cfg.data,
        normalize=True,
        is_distributed=cfg.trainer.gpus > 1
    )

    model = DeepSpeech(
        labels=labels,
        model_cfg=cfg.model,
        optim_cfg=cfg.optim,
        precision=cfg.trainer.precision,
        spect_cfg=cfg.data.spect
    )

    # Init Lightning Trainer
    trainer = hydra.utils.instantiate(
        config=cfg.trainer,
        replace_sampler_ddp=False,
        callbacks=[checkpoint_callback] if cfg.trainer.checkpoint_callback else None,
    )
    trainer.fit(model, data_loader)
