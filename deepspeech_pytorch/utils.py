import torch

from deepspeech_pytorch.decoder import GreedyDecoder
from deepspeech_pytorch.model import DeepSpeech


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
