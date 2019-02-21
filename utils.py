import torch
from apex.fp16_utils import BN_convert_float
import torch.distributed as dist

from model import DeepSpeech


def convert_model_to_half(model):
    """
    Converts model to half but keeps the batch norm layers in 32 bit for precision purposes
    """
    old_model = model
    new_model = BN_convert_float(model.half())
    del old_model  # Delete previous non-half model
    return new_model


def reduce_tensor(tensor, world_size, reduce_op_max=False):
    rt = tensor.clone()
    # Default to sum
    dist.all_reduce(
        rt, op=dist.reduce_op.MAX if reduce_op_max is True else dist.reduce_op.SUM)
    if not reduce_op_max:
        rt /= world_size
    return rt


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


def load_model(device, model_path, is_cuda):
    model = DeepSpeech.load_model(model_path)
    model.eval()
    model = model.to(device)
    if is_cuda and model.mixed_precision:
        model = convert_model_to_half(model)
    return model
