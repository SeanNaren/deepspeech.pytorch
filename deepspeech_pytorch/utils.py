import os
from pathlib import Path

import torch

from deepspeech_pytorch.decoder import GreedyDecoder
from deepspeech_pytorch.model import DeepSpeech


class CheckpointHandler:
    def __init__(self,
                 save_folder: str,
                 best_val_model_name: str,
                 checkpoint_per_iteration: int,
                 save_n_recent_models: int):
        self.save_folder = Path(save_folder)
        self.save_folder.mkdir(parents=True, exist_ok=True)  # Ensure save folder exists
        self.checkpoint_prefix = 'deepspeech_checkpoint_'  # TODO do we want to expose this?
        self.checkpoint_prefix_path = self.save_folder / self.checkpoint_prefix
        self.best_val_path = self.save_folder / best_val_model_name
        self.checkpoint_per_iteration = checkpoint_per_iteration
        self.save_n_recent_models = save_n_recent_models

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

    def save_checkpoint_model(self, epoch, state, i=None):
        if self.save_n_recent_models > 0:
            self.check_and_delete_oldest_checkpoint()
        model_path = self._create_checkpoint_path(epoch=epoch,
                                                  i=i)
        print("Saving checkpoint model to %s" % model_path)
        torch.save(obj=state.serialize_state(epoch=epoch,
                                             iteration=i),
                   f=model_path)

    def save_iter_checkpoint_model(self, epoch, state, i):
        if self.checkpoint_per_iteration > 0 and i > 0 and (i + 1) % self.checkpoint_per_iteration == 0:
            self.save_checkpoint_model(epoch=epoch,
                                       state=state,
                                       i=i)

    def save_best_model(self, epoch, state):
        print("Found better validated model, saving to %s" % self.best_val_path)
        torch.save(obj=state.serialize_state(epoch=epoch,
                                             iteration=None),
                   f=self.best_val_path)

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


def check_loss(loss, loss_value):
    """
    Check that warp-ctc loss is valid and will not break training
    :return: Return if loss is valid, and the error in case it is not
    """
    loss_valid = True
    error = ''
    if loss_value == float("inf") or loss_value == float("-inf"):
        loss_valid = False
        error = "WARNING: received an inf loss"
    elif torch.isnan(loss).sum() > 0:
        loss_valid = False
        error = 'WARNING: received a nan loss, setting loss value to 0'
    elif loss_value < 0:
        loss_valid = False
        error = "WARNING: received a negative loss"
    return loss_valid, error


def load_model(device,
               model_path,
               use_half):
    model = DeepSpeech.load_model(model_path)
    model.eval()
    model = model.to(device)
    if use_half:
        model = model.half()
    return model


def load_decoder(decoder_type,
                 labels,
                 lm_path,
                 alpha,
                 beta,
                 cutoff_top_n,
                 cutoff_prob,
                 beam_width,
                 lm_workers):
    if decoder_type == "beam":
        from deepspeech_pytorch.decoder import BeamCTCDecoder

        decoder = BeamCTCDecoder(labels=labels,
                                 lm_path=lm_path,
                                 alpha=alpha,
                                 beta=beta,
                                 cutoff_top_n=cutoff_top_n,
                                 cutoff_prob=cutoff_prob,
                                 beam_width=beam_width,
                                 num_processes=lm_workers)
    else:
        decoder = GreedyDecoder(labels=labels,
                                blank_index=labels.index('_'))
    return decoder


def remove_parallel_wrapper(model):
    """
    Return the model or extract the model out of the parallel wrapper
    :param model: The training model
    :return: The model without parallel wrapper
    """
    # Take care of distributed/data-parallel wrapper
    model_no_wrapper = model.module if hasattr(model, "module") else model
    return model_no_wrapper
