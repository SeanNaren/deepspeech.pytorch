import pytorch_lightning as pl
from hydra.utils import to_absolute_path

from deepspeech_pytorch.configs.train_config import DataConfig
from deepspeech_pytorch.loader.data_loader import SpectrogramDataset, DSRandomSampler, AudioDataLoader


class DeepSpeechDataModule(pl.LightningDataModule):

    def __init__(self,
                 labels,
                 data_cfg: DataConfig,
                 normalize,
                 epoch,
                 training_step):
        super().__init__()
        self.train_manifest = to_absolute_path(data_cfg.train_manifest)
        self.val_manifest = to_absolute_path(data_cfg.val_manifest)
        self.labels = labels
        self.data_cfg = data_cfg
        self.spect_cfg = data_cfg.spect
        self.aug_cfg = data_cfg.augmentation
        self.normalize = normalize
        self.epoch = epoch
        self.training_step = training_step

    def train_dataloader(self):
        train_dataset = self._create_dataset(self.train_manifest)
        train_sampler = DSRandomSampler(
            dataset=train_dataset,
            batch_size=self.data_cfg.batch_size,
            start_index=self.training_step
        )
        train_sampler.set_epoch(self.epoch)
        train_loader = AudioDataLoader(
            dataset=train_dataset,
            num_workers=self.data_cfg.num_workers,
            batch_sampler=train_sampler
        )
        return train_loader

    def val_dataloader(self):
        val_dataset = self._create_dataset(self.val_manifest)
        val_loader = AudioDataLoader(
            dataset=val_dataset,
            num_workers=self.data_cfg.num_workers,
            batch_size=self.data_cfg.batch_size
        )
        return val_loader

    def _create_dataset(self, manifest_filepath):
        dataset = SpectrogramDataset(
            audio_conf=self.spect_cfg,
            manifest_filepath=manifest_filepath,
            labels=self.labels,
            normalize=True,
            aug_cfg=self.aug_cfg
        )
        return dataset
