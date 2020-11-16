# deepspeech.pytorch
[![Build Status](https://travis-ci.org/SeanNaren/deepspeech.pytorch.svg?branch=master)](https://travis-ci.org/SeanNaren/deepspeech.pytorch)

Implementation of DeepSpeech2 for PyTorch. The repo supports training/testing and inference using the [DeepSpeech2](http://arxiv.org/pdf/1512.02595v1.pdf) model. Optionally a [kenlm](https://github.com/kpu/kenlm) language model can be used at inference time.

## Installation

### Docker

To use the image with a GPU you'll need to have [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) installed.

```bash
sudo docker run -ti --gpus all -v `pwd`/data:/workspace/data --tmpfs /tmp -p 8888:8888 --net=host --ipc=host seannaren/deepspeech.pytorch:latest # Opens a Jupyter notebook, mounting the /data drive in the container
```

Optionally you can use the command line by changing the entrypoint:

```bash
sudo docker run -ti --gpus all -v `pwd`/data:/workspace/data --tmpfs /tmp --entrypoint=/bin/bash --net=host --ipc=host seannaren/deepspeech.pytorch:latest
```

### From Source

Several libraries are needed to be installed for training to work. I will assume that everything is being installed in
an Anaconda installation on Ubuntu, with PyTorch installed.

Install [PyTorch](https://github.com/pytorch/pytorch#installation) if you haven't already.

Install this fork for Warp-CTC bindings:
```
git clone https://github.com/SeanNaren/warp-ctc.git
cd warp-ctc; mkdir build; cd build; cmake ..; make
export CUDA_HOME="/usr/local/cuda"
cd ../pytorch_binding && python setup.py install
```

Install NVIDIA apex:
```
git clone --recursive https://github.com/NVIDIA/apex.git
cd apex && pip install .
```

If you want decoding to support beam search with an optional language model, install ctcdecode:
```
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode && pip install .
```

Finally clone this repo and run this within the repo:
```
pip install -r requirements.txt
pip install -e . # Dev install
```

If you plan to use Multi-GPU/Multi-node training, you'll need etcd. Below is the command to install on Ubuntu.
```
sudo apt-get install etcd
```

## Training

### Datasets

Currently supports AN4, TEDLIUM, Voxforge, Common Voice and LibriSpeech. Scripts will setup the dataset and create manifest files used in data-loading. The scripts can be found in the data/ folder. Many of the scripts allow you to download the raw datasets separately if you choose so.

#### Custom Dataset

To create a custom dataset you must create a CSV file containing the locations of the training data. This has to be in the format of:

```
/path/to/audio.wav,/path/to/text.txt
/path/to/audio2.wav,/path/to/text2.txt
...
```

The first path is to the audio file, and the second path is to a text file containing the transcript on one line. This can then be used as stated below.


#### Merging multiple manifest files

To create bigger manifest files (to train/test on multiple datasets at once) we can merge manifest files together like below from a directory
containing all the manifests you want to merge. You can also prune short and long clips out of the new manifest.

```
cd data/
python merge_manifests.py --output-path merged_manifest.csv --merge-dir all-manifests/ --min-duration 1 --max-duration 15 # durations in seconds
```

### Training a Model

Configuration is done via [Hydra](https://github.com/facebookresearch/hydra).

Defaults can be seen in [config.py](deepspeech_pytorch/configs/train_config.py). Below is how you can override values set already:

```
python train.py data.train_manifest=data/train_manifest.csv data.val_manifest=data/val_manifest.csv
```

Use `python train.py --help` for all parameters and options.

You can also specify a config file to keep parameters stored in a yaml file like so:

Create folder `experiment/` and file `experiment/an4.yaml`:
```yaml
data:
  train_manifest: data/an4_train_manifest.csv
  val_manifest: data/an4_val_manifest.csv
```

```
python train.py +experiment=an4
```

There is also [Visdom](https://github.com/facebookresearch/visdom) support to visualize training. Once a server has been started, to use:

```
python train.py visualization.visdom=true
```

There is also Tensorboard support to visualize training. Follow the instructions to set up. To use:

```
python train.py visualization.tensorboard=true visualization.log_dir=log_dir/ # Make sure the Tensorboard instance is made pointing to this log directory
```

For both visualisation tools, you can add your own name to the run by changing the `--id` parameter when training.

### Multi-GPU Training

We support multi-GPU training via [TorchElastic](https://pytorch.org/elastic/0.2.0/index.html).

Below is an example command when training on a machine with 4 local GPUs:

```
python -m torchelastic.distributed.launch \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=4 \
        train.py data.train_manifest=data/an4_train_manifest.csv \
                 data.val_manifest=data/an4_val_manifest.csv  apex.opt_level=O1 data.num_workers=8 \
                 data.batch_size=8 training.epochs=70 checkpointing.checkpoint=true checkpointing.save_n_recent_models=3
```

You'll see the output for all the processes running on each individual GPU.
You can verify the model is being synchronized by the WER from all workers at validation time.

### Multi-Node Training

Also supported is multi-machine capabilities using TorchElastic. This requires a node to exist as an explicit etcd host (which could be one of the GPU nodes but isn't recommended), a shared mount across your cluster to load/save checkpoints and communication between the nodes.

Below is an example where we've set one of our GPU nodes as our etcd host however if you're scaling up, it would be suggested to have a separate instance as your etcd instance to your GPU nodes as this will be a single point of failure.

Assumed below is a shared drive called /share where we save our checkpoints and data to access.

Run on the etcd host:
```
PUBLIC_HOST_NAME=127.0.0.1 # Change to public host name for all nodes to connect
etcd --enable-v2 \
     --listen-client-urls http://$PUBLIC_HOST_NAME:4377 \
     --advertise-client-urls http://$PUBLIC_HOST_NAME:4377 \
     --listen-peer-urls http://$PUBLIC_HOST_NAME:4379
```

Run on each GPU node:
```
python -m torchelastic.distributed.launch \
        --nnodes=2 \
        --nproc_per_node=4 \
        --rdzv_id=123 \
        --rdzv_backend=etcd \
        --rdzv_endpoint=$PUBLIC_HOST_NAME:4377 \
        train.py data.train_manifest=/share/data/an4_train_manifest.csv \
                 data.val_manifest=/share/data/an4_val_manifest.csv apex.opt_level=O1 \
                 data.num_workers=8 checkpointing.save_folder=/share/checkpoints/ \
                 checkpointing.checkpoint=true checkpointing.load_auto_checkpoint=true checkpointing.save_n_recent_models=3 \
                 data.batch_size=8 training.epochs=70 
```

Using the `checkpointing.load_auto_checkpoint=true` flag and the `checkpointing.checkpoint_per_iteration` flag we can re-continue training from the latest saved checkpoint.

Currently it is expected that there is an NFS drive/shared mount across all nodes within the cluster to load the latest checkpoint from.

### Mixed Precision

If you are using NVIDIA volta cards or above to train your model, it's highly suggested to turn on mixed precision for speed/memory benefits. More information can be found [here](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html).

Different Optimization levels are available. More information on the Nvidia Apex API can be seen [here](https://nvidia.github.io/apex/amp.html#opt-levels).

```
python train.py data.train_manifest=data/train_manifest.csv data.val_manifest=data/val_manifest.csv apex.opt_level=O1 apex.loss_scale=1.0
```

Training a model in mixed-precision means you can use 32 bit float or half precision at runtime. Float32 is default, to use half precision (Which on V100s come with a speedup and better memory use) use the `--half` flag when testing or transcribing.

### Swapping to ADAMW Optimizer

ADAMW may provide better convergence and stability when training than SGD. In the future this may replace SGD within this repo.

```
python train.py data.train_manifest=data/train_manifest.csv data.val_manifest=data/val_manifest.csv optim=adam 
```

### Augmentation

There is support for three different types of augmentations: SpecAugment, noise injection and random tempo/gain perturbations.

#### SpecAugment

Applies simple Spectral Augmentation techniques directly on Mel spectogram features to make the model more robust to variations in input data. To enable SpecAugment, use the `--spec-augment` flag when training.

SpecAugment implementation was adapted from [this](https://github.com/DemisEom/SpecAugment) project.

#### Noise Injection

Dynamically adds noise into the training data to increase robustness. To use, first fill a directory up with all the noise files you want to sample from.
The dataloader will randomly pick samples from this directory.

To enable noise injection, use the `--noise-dir /path/to/noise/dir/` to specify where your noise files are. There are a few noise parameters to tweak, such as
`--noise_prob` to determine the probability that noise is added, and the `--noise-min`, `--noise-max` parameters to determine the minimum and maximum noise to add in training.

Included is a script to inject noise into an audio file to hear what different noise levels/files would sound like. Useful for curating the noise dataset.

```
python noise_inject.py --input-path /path/to/input.wav --noise-path /path/to/noise.wav --output-path /path/to/input_injected.wav --noise-level 0.5 # higher levels means more noise
```

#### Tempo/Gain Perturbation

Applies small changes to the tempo and gain when loading audio to increase robustness. To use, use the `--speed-volume-perturb` flag when training.

### Checkpoints

Training supports saving checkpoints of the model to continue training from should an error occur or early termination. To enable epoch
checkpoints use:

```
python train.py checkpoint=true
```

To enable checkpoints every N batches through the epoch as well as epoch saving:

```
python train.py checkpoint=true --checkpoint-per-batch N # N is the number of batches to wait till saving a checkpoint at this batch.
```

Note for the batch checkpointing system to work, you cannot change the batch size when loading a checkpointed model from it's original training
run.

To continue from a checkpointed model that has been saved:

```
python train.py checkpointing.continue_from=models/deepspeech_checkpoint_epoch_N_iter_N.pth
```

This continues from the same training state as well as recreates the visdom graph to continue from if enabled.

If you would like to start from a previous checkpoint model but not continue training, add the `training.finetune=true` flag to restart training
from the `checkpointing.continue_from` weights.

### Choosing batch sizes

Included is a script that can be used to benchmark whether training can occur on your hardware, and the limits on the size of the model/batch
sizes you can use. To use:

```
python benchmark.py --batch-size 32
```

Use the flag `--help` to see other parameters that can be used with the script.

To also note, there is no final softmax layer on the model as when trained, warp-ctc does this softmax internally. This will have to also be implemented in complex decoders if anything is built on top of the model, so take this into consideration!

## Testing/Inference

To evaluate a trained model on a test set (has to be in the same format as the training set):

```
python test.py model.model_path=models/deepspeech.pth test_manifest=/path/to/test_manifest.csv
```

An example script to output a transcription has been provided:

```
python transcribe.py model.model_path=models/deepspeech.pth audio_path=/path/to/audio.wav
```

If you used mixed-precision or half precision when training the model, you can use the `--half` flag for a speed/memory benefit.

## Inference Server

Included is a basic server script that will allow post request to be sent to the server to transcribe files.

```
python server.py --host 0.0.0.0 --port 8000 # Run on one window

curl -X POST http://0.0.0.0:8000/transcribe -H "Content-type: multipart/form-data" -F "file=@/path/to/input.wav"
```

## Using an ARPA LM

We support using kenlm based LMs. Below are instructions on how to take the LibriSpeech LMs found [here](http://www.openslr.org/11/) and tune the model to give you the best parameters when decoding, based on LibriSpeech.

### Tuning the LibriSpeech LMs

First ensure you've set up the librispeech datasets from the data/ folder.
In addition download the latest pre-trained librispeech model from the releases page, as well as the ARPA model you want to tune from [here](http://www.openslr.org/11/). For the below we use the 3-gram ARPA model (3e-7 prune).

First we need to generate the acoustic output to be used to evaluate the model on LibriSpeech val.
```
python test.py data.test_manifest=data/librispeech_val_manifest.csv model.model_path=librispeech_pretrained_v2.pth save_output=librispeech_val_output.npy
```

We use a beam width of 128 which gives reasonable results. We suggest using a CPU intensive node to carry out the grid search.

```
python search_lm_params.py --num-workers 16 --saved-output librispeech_val_output.npy --output-path libri_tune_output.json --lm-alpha-from 0 --lm-alpha-to 5 --lm-beta-from 0 --lm-beta-to 3 --lm-path 3-gram.pruned.3e-7.arpa  --model-path librispeech_pretrained_v2.pth --beam-width 128 --lm-workers 16
```

This will run a grid search across the alpha/beta parameters using a beam width of 128. Use the below script to find the best alpha/beta params:

```
python select_lm_params.py --input-path libri_tune_output.json
```

Use the alpha/beta parameters when using the beam decoder.

### Building your own LM

To build your own LM you need to use the KenLM repo found [here](https://github.com/kpu/kenlm). Have a read of the documentation to get a sense of how to train your own LM. The above steps once trained can be used to find the appropriate parameters.

### Alternate Decoders
By default, `test.py` and `transcribe.py` use a `GreedyDecoder` which picks the highest-likelihood output label at each timestep. Repeated and blank symbols are then filtered to give the final output.

A beam search decoder can optionally be used with the installation of the `ctcdecode` library as described in the Installation section. The `test` and `transcribe` scripts have a `decoder_type` argument. To use the beam decoder, add `lm.decoder_type=beam`. The beam decoder enables additional decoding parameters:
- **lm.beam_width** how many beams to consider at each timestep
- **lm.lm_path** optional binary KenLM language model to use for decoding
- **lm.alpha** weight for language model
- **lm.beta** bonus weight for words

### Time offsets

Use the `--offsets` flag to get positional information of each character in the transcription when using `transcribe.py` script. The offsets are based on the size
of the output tensor, which you need to convert into a format required.
For example, based on default parameters you could multiply the offsets by a scalar (duration of file in seconds / size of output) to get the offsets in seconds.

## Pre-trained models

Pre-trained models can be found under releases [here](https://github.com/SeanNaren/deepspeech.pytorch/releases).

## Acknowledgements

Thanks to [Egor](https://github.com/EgorLakomkin) and [Ryan](https://github.com/ryanleary) for their contributions!
