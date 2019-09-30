# deepspeech.pytorch

Implementation of DeepSpeech2 for PyTorch. Creates a network based on the [DeepSpeech2](http://arxiv.org/pdf/1512.02595v1.pdf) architecture, trained with the CTC activation function.

## Installation

### Docker

There is no official Dockerhub image, however a Dockerfile is provided to build on your own systems.

```bash
sudo nvidia-docker build -t  deepspeech2.docker .
sudo nvidia-docker run -ti -v `pwd`/data:/workspace/data -p 8888:8888 --net=host --ipc=host deepspeech2.docker # Opens a Jupyter notebook, mounting the /data drive in the container
```

Optionally you can use the command line by changing the entrypoint:

```bash
sudo nvidia-docker run -ti -v `pwd`/data:/workspace/data --entrypoint=/bin/bash --net=host --ipc=host deepspeech2.docker

```

### From Source

Several libraries are needed to be installed for training to work. I will assume that everything is being installed in
an Anaconda installation on Ubuntu, with Pytorch 1.0.

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

```
python train.py --train-manifest data/train_manifest.csv --val-manifest data/val_manifest.csv
```

Use `python train.py --help` for more parameters and options.

There is also [Visdom](https://github.com/facebookresearch/visdom) support to visualize training. Once a server has been started, to use:

```
python train.py --visdom
```

There is also [Tensorboard](https://github.com/lanpa/tensorboard-pytorch) support to visualize training. Follow the instructions to set up. To use:

```
python train.py --tensorboard --logdir log_dir/ # Make sure the Tensorboard instance is made pointing to this log directory
```

For both visualisation tools, you can add your own name to the run by changing the `--id` parameter when training.

### Multi-GPU Training

We support multi-GPU training via the distributed parallel wrapper (see [here](https://github.com/NVIDIA/sentiment-discovery/blob/master/analysis/scale.md) and [here](https://github.com/SeanNaren/deepspeech.pytorch/issues/211) to see why we don't use DataParallel).

To use multi-GPU:

```
python -m multiproc train.py --visdom --cuda # Add your parameters as normal, multiproc will scale to all GPUs automatically
```

multiproc will open a log for all processes other than the main process.

You can also specify specific GPU IDs rather than allowing the script to use all available GPUs:

```
python -m multiproc train.py --visdom --cuda --device-ids 0,1,2,3 # Add your parameters as normal, will only run on 4 GPUs
```

We suggest using the NCCL backend which defaults to TCP if Infiniband isn't available.

### Mixed Precision

If you are using NVIDIA volta cards or above to train your model, it's highly suggested to turn on mixed precision for speed/memory benefits. More information can be found [here](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html).

Different Optimization levels are available. More information on the Nvidia Apex API can be seen [here](https://nvidia.github.io/apex/amp.html#opt-levels).

```
python train.py --train-manifest data/train_manifest.csv --val-manifest data/val_manifest.csv --opt-level O1 --loss-scale 1.0
```

Training a model in mixed-precision means you can use 32 bit float or half precision at runtime. Float is default, to use half precision (Which on V100s come with a speedup and better memory use) use the `--half` flag when testing or transcribing.

### Noise Augmentation/Injection

There is support for two different types of noise; noise augmentation and noise injection.

#### Noise Augmentation

Applies small changes to the tempo and gain when loading audio to increase robustness. To use, use the `--augment` flag when training.

#### Noise Injection

Dynamically adds noise into the training data to increase robustness. To use, first fill a directory up with all the noise files you want to sample from.
The dataloader will randomly pick samples from this directory.

To enable noise injection, use the `--noise-dir /path/to/noise/dir/` to specify where your noise files are. There are a few noise parameters to tweak, such as
`--noise_prob` to determine the probability that noise is added, and the `--noise-min`, `--noise-max` parameters to determine the minimum and maximum noise to add in training.

Included is a script to inject noise into an audio file to hear what different noise levels/files would sound like. Useful for curating the noise dataset.

```
python noise_inject.py --input-path /path/to/input.wav --noise-path /path/to/noise.wav --output-path /path/to/input_injected.wav --noise-level 0.5 # higher levels means more noise
```

### Checkpoints

Training supports saving checkpoints of the model to continue training from should an error occur or early termination. To enable epoch
checkpoints use:

```
python train.py --checkpoint
```

To enable checkpoints every N batches through the epoch as well as epoch saving:

```
python train.py --checkpoint --checkpoint-per-batch N # N is the number of batches to wait till saving a checkpoint at this batch.
```

Note for the batch checkpointing system to work, you cannot change the batch size when loading a checkpointed model from it's original training
run.

To continue from a checkpointed model that has been saved:

```
python train.py --continue-from models/deepspeech_checkpoint_epoch_N_iter_N.pth
```

This continues from the same training state as well as recreates the visdom graph to continue from if enabled.

If you would like to start from a previous checkpoint model but not continue training, add the `--finetune` flag to restart training
from the `--continue-from` weights.

### Choosing batch sizes

Included is a script that can be used to benchmark whether training can occur on your hardware, and the limits on the size of the model/batch
sizes you can use. To use:

```
python benchmark.py --batch-size 32
```

Use the flag `--help` to see other parameters that can be used with the script.

### Model details

Saved models contain the metadata of their training process. To see the metadata run the below command:

```
python model.py --model-path models/deepspeech.pth
```

To also note, there is no final softmax layer on the model as when trained, warp-ctc does this softmax internally. This will have to also be implemented in complex decoders if anything is built on top of the model, so take this into consideration!

## Testing/Inference

To evaluate a trained model on a test set (has to be in the same format as the training set):

```
python test.py --model-path models/deepspeech.pth --test-manifest /path/to/test_manifest.csv --cuda
```

An example script to output a transcription has been provided:

```
python transcribe.py --model-path models/deepspeech.pth --audio-path /path/to/audio.wav
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
python test.py --test-manifest data/librispeech_val_manifest.csv --model-path librispeech_pretrained_v2.pth --cuda --half --save-output librispeech_val_output.npy
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

A beam search decoder can optionally be used with the installation of the `ctcdecode` library as described in the Installation section. The `test` and `transcribe` scripts have a `--decoder` argument. To use the beam decoder, add `--decoder beam`. The beam decoder enables additional decoding parameters:
- **beam_width** how many beams to consider at each timestep
- **lm_path** optional binary KenLM language model to use for decoding
- **alpha** weight for language model
- **beta** bonus weight for words

### Time offsets

Use the `--offsets` flag to get positional information of each character in the transcription when using `transcribe.py` script. The offsets are based on the size
of the output tensor, which you need to convert into a format required.
For example, based on default parameters you could multiply the offsets by a scalar (duration of file in seconds / size of output) to get the offsets in seconds.

## Pre-trained models

Pre-trained models can be found under releases [here](https://github.com/SeanNaren/deepspeech.pytorch/releases).

## Acknowledgements

Thanks to [Egor](https://github.com/EgorLakomkin) and [Ryan](https://github.com/ryanleary) for their contributions!
