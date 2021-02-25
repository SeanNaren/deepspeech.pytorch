import os
from pathlib import Path

import hydra
from google.cloud import storage
from hydra_configs.pytorch_lightning.callbacks import ModelCheckpointConf
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm

from deepspeech_pytorch.configs.train_config import GCSCheckpointConfig


class CheckpointHandler(ModelCheckpoint):

    def __init__(self, cfg: ModelCheckpointConf):
        super().__init__(
            dirpath=cfg.dirpath,
            filename=cfg.filename,
            monitor=cfg.monitor,
            verbose=cfg.verbose,
            save_last=cfg.save_last,
            save_top_k=cfg.save_top_k,
            save_weights_only=cfg.save_weights_only,
            mode=cfg.mode,
            period=cfg.period,
            prefix=cfg.prefix
        )

    def find_latest_checkpoint(self):
        raise NotImplementedError


class FileCheckpointHandler(CheckpointHandler):

    def find_latest_checkpoint(self):
        """
        Finds the latest checkpoint in a folder based on the timestamp of the file.
        If there are no checkpoints, returns None.
        :return: The latest checkpoint path, or None if no checkpoints are found.
        """
        paths = list(Path(self.dirpath).rglob(self.prefix + '*'))
        if paths:
            paths.sort(key=os.path.getctime)
            latest_checkpoint_path = paths[-1]
            return latest_checkpoint_path
        else:
            return None


class GCSCheckpointHandler(CheckpointHandler):
    def __init__(self, cfg: GCSCheckpointConfig):
        self.client = storage.Client()
        self.gcs_bucket = cfg.gcs_bucket
        self.gcs_save_folder = cfg.gcs_save_folder
        self.bucket = self.client.bucket(bucket_name=self.gcs_bucket)
        super().__init__(cfg=cfg)

    def find_latest_checkpoint(self):
        """
        Finds the latest checkpoint in a folder based on the timestamp of the file.
        Downloads the GCS checkpoint to a local file, and returns the local file path.
        If there are no checkpoints, returns None.
        :return: The latest checkpoint path, or None if no checkpoints are found.
        """
        prefix = self.gcs_save_folder + self.prefix
        paths = list(self.client.list_blobs(self.gcs_bucket, prefix=prefix))
        if paths:
            paths.sort(key=lambda x: x.time_created)
            latest_blob = paths[-1]
            latest_blob.download_to_filename(self.local_save_file)
            return self.local_save_file
        else:
            return None

    def _save_model(self, filepath: str, trainer, pl_module):

        # in debugging, track when we save checkpoints
        trainer.dev_debugger.track_checkpointing_history(filepath)

        # make paths
        if trainer.is_global_zero:
            tqdm.write(f"Saving model to {filepath}")
            trainer.save_checkpoint(filepath)
            self._save_file_to_gcs(filepath, self.save_weights_only)

    def _save_file_to_gcs(self, model_path):
        tqdm.write(f"Saving model to gs://{self.gcs_bucket}/{self.gcs_save_folder}/{self.filename}{self.FILE_EXTENSION}")
        blob = self.bucket.blob(f"{self.gcs_save_folder}/{self.filename}{self.FILE_EXTENSION}")
        blob.upload_from_filename(model_path)
