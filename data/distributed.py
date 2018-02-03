import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch.distributed as dist
from torch.nn.modules import Module

'''
This version of DistributedDataParallel is designed to be used in conjunction with the multiproc.py
launcher included with this example. It assumes that your run is using multiprocess with 1
GPU/process, that the model is on the correct device, and that torch.set_device has been
used to set the device.

Parameters are broadcasted to the other processes on initialization of DistributedDataParallel,
and will be allreduced at the finish of the backward pass.
'''


class DistributedDataParallel(Module):
    def __init__(self, module):
        super(DistributedDataParallel, self).__init__()
        self.warn_on_half = True if dist._backend == dist.dist_backend.GLOO else False

        self.module = module

        for p in self.module.state_dict().values():
            if not torch.is_tensor(p):
                continue
            if dist._backend == dist.dist_backend.NCCL:
                assert p.is_cuda, "NCCL backend only supports model parameters to be on GPU."
            dist.broadcast(p, 0)

        def allreduce_params():
            if (self.needs_reduction):
                self.needs_reduction = False
                buckets = {}
                for param in self.module.parameters():
                    if param.requires_grad and param.grad is not None:
                        tp = type(param.data)
                        if tp not in buckets:
                            buckets[tp] = []
                        buckets[tp].append(param)
                if self.warn_on_half:
                    if torch.cuda.HalfTensor in buckets:
                        print("WARNING: gloo dist backend for half parameters may be extremely slow." +
                              " It is recommended to use the NCCL backend in this case.")
                        self.warn_on_half = False

                for tp in buckets:
                    bucket = buckets[tp]
                    grads = [param.grad.data for param in bucket]
                    coalesced = _flatten_dense_tensors(grads)
                    dist.all_reduce(coalesced)
                    coalesced /= dist.get_world_size()
                    for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                        buf.copy_(synced)

        for param in list(self.module.parameters()):
            def allreduce_hook(*unused):
                param._execution_engine.queue_callback(allreduce_params)

            if param.requires_grad:
                param.register_hook(allreduce_hook)

    def forward(self, *inputs, **kwargs):
        self.needs_reduction = True
        return self.module(*inputs, **kwargs)
