import os
from abc import ABC
from pathlib import Path, PosixPath

import hydra
import torch
from deepspeech_pytorch.configs.train_config import GCSCheckpointConfig, CheckpointConfig, FileCheckpointConfig
from deepspeech_pytorch.state import TrainingState
from google.cloud import storage


class CheckpointHandler(ABC):

    def __init__(self,
                 cfg: CheckpointConfig,
                 save_location):
        self.checkpoint_prefix = 'deepspeech_checkpoint_'  # TODO do we want to expose this?
        self.save_location = save_location
        self.checkpoint_per_iteration = cfg.checkpoint_per_iteration
        self.save_n_recent_models = cfg.save_n_recent_models

        if type(self.save_location) == PosixPath:
            self.checkpoint_prefix_path = self.save_location / self.checkpoint_prefix
            self.best_val_path = self.save_location / cfg.best_val_model_name
        else:
            self.checkpoint_prefix_path = self.save_location + self.checkpoint_prefix
            self.best_val_path = self.save_location + cfg.best_val_model_name

    def save_model(self,
                   model_path: str,
                   state: TrainingState,
                   epoch: int,
                   i: int = None):
        raise NotImplementedError

    def find_latest_checkpoint(self):
        raise NotImplementedError

    def check_and_delete_oldest_checkpoint(self):
        raise NotImplementedError

    def save_checkpoint_model(self, epoch, state, i=None):
        if self.save_n_recent_models > 0:
            self.check_and_delete_oldest_checkpoint()
        model_path = self._create_checkpoint_path(epoch=epoch,
                                                  i=i)
        self.save_model(model_path=model_path,
                        state=state,
                        epoch=epoch,
                        i=i)

    def save_iter_checkpoint_model(self, epoch, state, i):
        if self.checkpoint_per_iteration > 0 and i > 0 and (i + 1) % self.checkpoint_per_iteration == 0:
            self.save_checkpoint_model(epoch=epoch,
                                       state=state,
                                       i=i)

    def save_best_model(self, epoch, state):
        self.save_model(model_path=self.best_val_path,
                        state=state,
                        epoch=epoch)

    def _create_checkpoint_path(self, epoch, i=None):
        """
        Creates path to save checkpoint.
        We automatically iterate the epoch and iteration for readibility.
        :param epoch: The epoch (index starts at 0).
        :param i: The iteration (index starts at 0).
        :return: The path to save the model
        """
        if i:
            checkpoint_path = str(self.checkpoint_prefix_path) + 'epoch_%d_iter_%d.pth' % (epoch + 1, i + 1)
        else:
            checkpoint_path = str(self.checkpoint_prefix_path) + 'epoch_%d.pth' % (epoch + 1)
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
            print("Deleting old checkpoint %s" % str(paths[0]))
            os.remove(paths[0])

    def save_model(self, model_path, state, epoch, i=None):
        print("Saving model to %s" % model_path)
        torch.save(obj=state.serialize_state(epoch=epoch,
                                             iteration=i),
                   f=model_path)


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
            print("Deleting old checkpoint %s" % paths[0].name)
            paths[0].delete()

    def save_model(self, model_path, state, epoch, i=None):
        print("Saving model to %s" % model_path)
        torch.save(obj=state.serialize_state(epoch=epoch,
                                             iteration=i),
                   f=self.local_save_file)
        self._save_file_to_gcs(model_path)

    def _save_file_to_gcs(self, model_path):
        blob = self.bucket.blob(model_path)
        blob.upload_from_filename(self.local_save_file)
