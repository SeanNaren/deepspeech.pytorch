import pytorch_lightning as pl

from deepspeech_pytorch.data.config import DataConfig
from deepspeech_pytorch.data.data_loader import SpectrogramDataset, DSRandomSampler, AudioDataLoader, \
    DSElasticDistributedSampler


class DeepSpeechDataModule(pl.LightningDataModule):

    def __init__(
            self,
            labels: list,
            config: DataConfig,
            normalize: bool = True):
        super().__init__()
        self.train_path = config.train_path
        self.val_path = config.val_path
        self.labels = labels
        self.config = config
        self.normalize = normalize

    def train_dataloader(self):
        train_dataset = self._create_dataset(self.train_path)

        if self.trainer.num_devices:
            train_sampler = DSElasticDistributedSampler(
                dataset=train_dataset,
                batch_size=self.config.batch_size
            )
        else:
            train_sampler = DSRandomSampler(
                dataset=train_dataset,
                batch_size=self.config.batch_size
            )
        return AudioDataLoader(
            dataset=train_dataset,
            num_workers=self.config.num_workers,
            batch_sampler=train_sampler
        )

    def val_dataloader(self):
        val_dataset = self._create_dataset(self.val_path)
        return AudioDataLoader(
            dataset=val_dataset,
            num_workers=self.config.num_workers,
            batch_size=self.config.batch_size
        )

    def _create_dataset(self, input_path):
        return SpectrogramDataset(
            audio_conf=self.config.spect,
            input_path=input_path,
            labels=self.labels,
            normalize=True,
            aug_cfg=self.config.augmentation
        )
