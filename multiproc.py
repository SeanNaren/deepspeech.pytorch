import os
import numpy as np
import torch
import sys
import subprocess

argslist = list(sys.argv)[1:]

world_size = int(os.environ.get('WORLD_SIZE', 1))
rank = int(os.environ.get('RANK', 0))

num_gpus = torch.cuda.device_count()
gpu_world_size = world_size * num_gpus  # We assume that each node has the same number of GPUs

ranks = list(range(gpu_world_size))  # Get all ranks before finding our specific ranks for GPUs
ranks = np.array_split(ranks, world_size)[rank]

argslist.append('--world_size')
argslist.append(str(gpu_world_size))

workers = []
for i in range(num_gpus):
    rank = str(ranks[i])
    gpu_rank = str(i)
    if '--rank' in argslist:
        argslist[argslist.index('--rank') + 1] = rank
    else:
        argslist.append('--rank')
        argslist.append(rank)
    if '--gpu_rank' in argslist:
        argslist[argslist.index('--gpu_rank') + 1] = gpu_rank
    else:
        argslist.append('--gpu_rank')
        argslist.append(gpu_rank)
    stdout = None if i == 0 else open("GPU_" + str(i) + ".log", "w")
    print(argslist)
    p = subprocess.Popen([str(sys.executable)] + argslist, stdout=stdout, stderr=stdout)
    workers.append(p)

for p in workers:
    return_code = p.wait()
    if return_code != 0:
        for p in workers:
            p.kill()  # Ensure all processes are killed
        raise RuntimeError('An error occurred when running a process')
