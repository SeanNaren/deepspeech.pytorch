import torch
import sys
import subprocess

argslist = list(sys.argv)[1:]
world_size = torch.cuda.device_count()
device_ids = None
if '--device-ids' in argslist:  # Manually specified GPU IDs
    device_ids = argslist[argslist.index('--device-ids') + 1].strip().split(',')
    world_size = len(device_ids)
    # Remove GPU IDs since these are not for the training script
    argslist.pop(argslist.index('--device-ids') + 1)
    argslist.pop(argslist.index('--device-ids'))

if '--world-size' in argslist:
    argslist[argslist.index('--world-size') + 1] = str(world_size)
else:
    argslist.append('--world-size')
    argslist.append(str(world_size))

workers = []

for i in range(world_size):
    if '--rank' in argslist:
        argslist[argslist.index('--rank') + 1] = str(i)
    else:
        argslist.append('--rank')
        argslist.append(str(i))
    if '--gpu-rank' in argslist:
        if device_ids:
            argslist[argslist.index('--gpu-rank') + 1] = str(device_ids[i])
        else:
            argslist[argslist.index('--gpu-rank') + 1] = str(i)
    else:
        argslist.append('--gpu-rank')
        argslist.append(str(i))
    stdout = None if i == 0 else open("GPU_" + str(i) + ".log", "w")
    print(argslist)
    p = subprocess.Popen([str(sys.executable)] + argslist, stdout=stdout, stderr=stdout)
    workers.append(p)

for p in workers:
    p.wait()
    if p.returncode != 0:
        raise subprocess.CalledProcessError(returncode=p.returncode,
                                            cmd=p.args)
