import os
from pathlib import Path, PosixPath

import hydra
from google.cloud import storage
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.utilities import rank_zero_only
from tqdm import tqdm

from deepspeech_pytorch.configs.train_config import GCSCheckpointConfig, CheckpointConfig, FileCheckpointConfig


class CheckpointHandler(Callback):

    def __init__(self,
                 cfg: CheckpointConfig,
                 save_location):
        self.cfg = cfg
        self.lowest_loss = None
        self.checkpoint_prefix = 'deepspeech_checkpoint_'  # TODO do we want to expose this?
        self.save_location = save_location
        self.save_n_recent_models = cfg.save_n_recent_models

        if type(self.save_location) == PosixPath:
            self.checkpoint_prefix_path = str(self.save_location / self.checkpoint_prefix)
            self.best_val_path = str(self.save_location / cfg.best_val_model_name)
        else:
            self.checkpoint_prefix_path = self.save_location + self.checkpoint_prefix
            self.best_val_path = self.save_location + cfg.best_val_model_name

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        if self.cfg.checkpoint:
            self.save_checkpoint_model(
                epoch=epoch,
                trainer=trainer
            )
        loss = metrics['loss']
        if self.lowest_loss is None or self.lowest_loss > loss:
            self.save_best_model(
                epoch=epoch,
                trainer=trainer
            )
            self.lowest_loss = loss

    def save_model(self,
                   model_path: str,
                   trainer: Trainer,
                   epoch: int):
        raise NotImplementedError

    def find_latest_checkpoint(self):
        raise NotImplementedError

    def check_and_delete_oldest_checkpoint(self):
        raise NotImplementedError

    def save_checkpoint_model(self,
                              epoch: int,
                              trainer: Trainer):
        if self.save_n_recent_models > 0:
            self.check_and_delete_oldest_checkpoint()
        model_path = self._create_checkpoint_path(
            epoch=epoch
        )
        self.save_model(
            model_path=model_path,
            epoch=epoch,
            trainer=trainer,
        )

    def save_best_model(self,
                        epoch: int,
                        trainer: Trainer):
        self.save_model(
            model_path=self.best_val_path,
            trainer=trainer,
            epoch=epoch
        )

    def _create_checkpoint_path(self, epoch):
        """
        Creates path to save checkpoint.
        :param epoch: The epoch.
        :return: The path to save the model
        """
        checkpoint_path = str(self.checkpoint_prefix_path) + 'epoch_%d.pth' % epoch
        return checkpoint_path


class FileCheckpointHandler(CheckpointHandler):
    def __init__(self, cfg: FileCheckpointConfig):
        self.save_folder = Path(hydra.utils.to_absolute_path(cfg.save_folder))
        self.save_folder.mkdir(parents=True, exist_ok=True)  # Ensure save folder exists
        super().__init__(cfg=cfg,
                         save_location=self.save_folder)

    def find_latest_checkpoint(self):
        """
        Finds the latest checkpoint in a folder based on the timestamp of the file.
        If there are no checkpoints, returns None.
        :return: The latest checkpoint path, or None if no checkpoints are found.
        """
        paths = list(self.save_folder.rglob(self.checkpoint_prefix + '*'))
        if paths:
            paths.sort(key=os.path.getctime)
            latest_checkpoint_path = paths[-1]
            return latest_checkpoint_path
        else:
            return None

    def check_and_delete_oldest_checkpoint(self):
        paths = list(self.save_folder.rglob(self.checkpoint_prefix + '*'))
        if paths and len(paths) >= self.save_n_recent_models:
            paths.sort(key=os.path.getctime)
            tqdm.write("Deleting old checkpoint %s" % str(paths[0]))
            os.remove(paths[0])

    def save_model(self,
                   model_path: str,
                   trainer: Trainer,
                   epoch: int,
                   i=None):
        tqdm.write("Saving model to %s" % model_path)
        trainer.save_checkpoint(model_path)


class GCSCheckpointHandler(CheckpointHandler):
    def __init__(self, cfg: GCSCheckpointConfig):
        self.client = storage.Client()
        self.local_save_file = hydra.utils.to_absolute_path(cfg.local_save_file)
        self.gcs_bucket = cfg.gcs_bucket
        self.bucket = self.client.bucket(bucket_name=self.gcs_bucket)
        super().__init__(cfg=cfg,
                         save_location=cfg.gcs_save_folder)

    def find_latest_checkpoint(self):
        """
        Finds the latest checkpoint in a folder based on the timestamp of the file.
        Downloads the GCS checkpoint to a local file, and returns the local file path.
        If there are no checkpoints, returns None.
        :return: The latest checkpoint path, or None if no checkpoints are found.
        """
        prefix = self.save_location + self.checkpoint_prefix
        paths = list(self.client.list_blobs(self.gcs_bucket, prefix=prefix))
        if paths:
            paths.sort(key=lambda x: x.time_created)
            latest_blob = paths[-1]
            latest_blob.download_to_filename(self.local_save_file)
            return self.local_save_file
        else:
            return None

    def check_and_delete_oldest_checkpoint(self):
        prefix = self.save_location + self.checkpoint_prefix
        paths = list(self.client.list_blobs(self.gcs_bucket, prefix=prefix))
        if paths and len(paths) >= self.save_n_recent_models:
            paths.sort(key=lambda x: x.time_created)
            tqdm.write("Deleting old checkpoint %s" % paths[0].name)
            paths[0].delete()

    def save_model(self,
                   model_path: str,
                   trainer: Trainer,
                   epoch: int):
        tqdm.write("Saving model to %s" % model_path)
        trainer.save_checkpoint(model_path)
        self._save_file_to_gcs(model_path)

    def _save_file_to_gcs(self, model_path):
        blob = self.bucket.blob(model_path)
        blob.upload_from_filename(self.local_save_file)
