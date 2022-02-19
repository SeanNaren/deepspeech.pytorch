# deepspeech.pytorch
![Tests](https://github.com/SeanNaren/deepspeech.pytorch/actions/workflows/ci-test.yml/badge.svg)

Implementation of DeepSpeech2 for PyTorch using [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning). The repo supports training/testing and inference using the [DeepSpeech2](http://arxiv.org/pdf/1512.02595v1.pdf) model. Optionally a [kenlm](https://github.com/kpu/kenlm) language model can be used at inference time.

## Install

Several libraries are needed to be installed for training to work. I will assume that everything is being installed in
an Anaconda installation on Ubuntu, with PyTorch installed.

Install [PyTorch](https://github.com/pytorch/pytorch#installation) if you haven't already.

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

If you plan to use Multi-node training, you'll need etcd. Below is the command to install on Ubuntu.
```
sudo apt-get install etcd
```

### Docker

To use the image with a GPU you'll need to have [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) installed.

```bash
sudo docker run -ti --gpus all -v `pwd`/data:/workspace/data --tmpfs /tmp -p 8888:8888 --net=host --ipc=host seannaren/deepspeech.pytorch:latest # Opens a Jupyter notebook, mounting the /data drive in the container
```

Optionally you can use the command line by changing the entrypoint:

```bash
sudo docker run -ti --gpus all -v `pwd`/data:/workspace/data --tmpfs /tmp --entrypoint=/bin/bash --net=host --ipc=host seannaren/deepspeech.pytorch:latest
```

## Training

### Datasets

Currently supports [AN4](http://www.speech.cs.cmu.edu/databases/an4/), [TEDLIUM](https://www.openslr.org/51/), [Voxforge](http://www.voxforge.org/), [Common Voice](https://commonvoice.mozilla.org/en/datasets) and [LibriSpeech](https://www.openslr.org/12). Scripts will setup the dataset and create manifest files used in data-loading. The scripts can be found in the data/ folder. Many of the scripts allow you to download the raw datasets separately if you choose so.

### Training Commands

##### AN4

```bash
cd data/ && python an4.py && cd ..

python train.py +configs=an4
```

##### LibriSpeech

```bash
cd data/ && python librispeech.py && cd ..

python train.py +configs=librispeech
```

##### Common Voice

```bash
cd data/ && python common_voice.py && cd ..

python train.py +configs=commonvoice
```
##### TEDlium

```bash
cd data/ && python ted.py && cd ..

python train.py +configs=tedlium
```

#### Custom Dataset

To create a custom dataset you must create a JSON file containing the locations of the training/testing data. This has to be in the format of:
```json
{
  "root_path":"path/to",
  "samples":[
    {"wav_path":"audio.wav","transcript_path":"text.txt"},
    {"wav_path":"audio2.wav","transcript_path":"text2.txt"},
    ...
  ]
}
```
Where the `root_path` is the root directory, `wav_path` is to the audio file, and the `transcript_path` is to a text file containing the transcript on one line. This can then be used as stated below.

##### Note on CSV files ...
Up until release [V2.1](https://github.com/SeanNaren/deepspeech.pytorch/releases/tag/V2.1), deepspeech.pytorch used CSV manifest files instead of JSON.
These manifest files are formatted similarly as a 2 column table:
```
/path/to/audio.wav,/path/to/text.txt
/path/to/audio2.wav,/path/to/text2.txt
...
```
Note that this format is incompatible [V3.0](https://github.com/SeanNaren/deepspeech.pytorch/releases/tag/V3.0) onwards.

#### Merging multiple manifest files

To create bigger manifest files (to train/test on multiple datasets at once) we can merge manifest files together like below.

```
cd data/
python merge_manifests.py manifest_1.json manifest_2.json --out new_manifest_dir
```

### Modifying Training Configs

Configuration is done via [Hydra](https://github.com/facebookresearch/hydra).

Defaults can be seen in [config.py](deepspeech_pytorch/configs/train_config.py). Below is how you can override values set already:

```
python train.py data.train_path=data/train_manifest.json data.val_path=data/val_manifest.json
```

Use `python train.py --help` for all parameters and options.

You can also specify a config file to keep parameters stored in a yaml file like so:

Create folder `experiment/` and file `experiment/an4.yaml`:
```yaml
data:
  train_path: data/an4_train_manifest.json
  val_path: data/an4_val_manifest.json
```

```
python train.py +experiment=an4
```

To see options available, check [here](./deepspeech_pytorch/configs/train_config.py).

### Multi-GPU Training

We support single-machine multi-GPU training via [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning).

Below is an example command when training on a machine with 4 local GPUs:

```
python train.py +configs=an4 trainer.gpus=4
```

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
        train.py data.train_path=/share/data/an4_train_manifest.json \
                 data.val_path=/share/data/an4_val_manifest.json model.precision=half \
                 data.num_workers=8 checkpoint.save_folder=/share/checkpoints/ \
                 checkpoint.checkpoint=true checkpoint.load_auto_checkpoint=true checkpointing.save_n_recent_models=3 \
                 data.batch_size=8 trainer.max_epochs=70 \
                 trainer.accelerator=ddp trainer.gpus=4 trainer.num_nodes=2
```

Using the `load_auto_checkpoint=true` flag we can re-continue training from the latest saved checkpoint.

Currently it is expected that there is an NFS drive/shared mount across all nodes within the cluster to load the latest checkpoint from.

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

Typically checkpoints are stored in `lightning_logs/` in the current working directory of the script.

This can be adjusted:

```
python train.py checkpoint.file_path=save_dir/
```

To load a previously saved checkpoint:

```
python train.py trainer.resume_from_checkpoint=lightning_logs/deepspeech_checkpoint_epoch_N_iter_N.ckpt
```

This continues from the same training state.

## Testing/Inference

To evaluate a trained model on a test set (has to be in the same format as the training set):

```
python test.py model.model_path=models/deepspeech.pth test_path=/path/to/test_manifest.json
```

An example script to output a transcription has been provided:

```
python transcribe.py model.model_path=models/deepspeech.pth audio_path=/path/to/audio.wav
```

If you used mixed-precision or half precision when training the model, you can use the `model.precision=half` for a speed/memory benefit.

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
python test.py data.test_path=data/librispeech_val_manifest.json model.model_path=librispeech_pretrained_v2.pth save_output=librispeech_val_output.npy
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

A beam search decoder can optionally be used with the installation of the `ctcdecode` library as described in the Installation section. The `test` and `transcribe` scripts have a `lm` config. To use the beam decoder, add `lm.decoder_type=beam`. The beam decoder enables additional decoding parameters:
- **lm.beam_width** how many beams to consider at each timestep
- **lm.lm_path** optional binary KenLM language model to use for decoding
- **lm.alpha** weight for language model
- **lm.beta** bonus weight for words

### Time offsets

Use the `offsets=true` flag to get positional information of each character in the transcription when using `transcribe.py` script. The offsets are based on the size
of the output tensor, which you need to convert into a format required.
For example, based on default parameters you could multiply the offsets by a scalar (duration of file in seconds / size of output) to get the offsets in seconds.

## Pre-trained models

Pre-trained models can be found under releases [here](https://github.com/SeanNaren/deepspeech.pytorch/releases).

## Acknowledgements

Thanks to [Egor](https://github.com/EgorLakomkin) and [Ryan](https://github.com/ryanleary) for their contributions!
