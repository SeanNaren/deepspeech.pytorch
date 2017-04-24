[![Stories in Ready](https://badge.waffle.io/SeanNaren/deepspeech.pytorch.png?label=ready&title=Ready)](http://waffle.io/SeanNaren/deepspeech.pytorch)
# deepspeech.pytorch

Implementation of DeepSpeech2 using [Baidu Warp-CTC](https://github.com/baidu-research/warp-ctc).
Creates a network based on the [DeepSpeech2](http://arxiv.org/pdf/1512.02595v1.pdf) architecture, trained with the CTC activation function.

# Installation

Several libraries are needed to be installed for training to work. I will assume that everything is being installed in
an Anaconda installation on Ubuntu.

Install [PyTorch](https://github.com/pytorch/pytorch#installation) if you haven't already.

Install this fork for Warp-CTC bindings:
```
git clone https://github.com/SeanNaren/warp-ctc.git
cd warp-ctc
mkdir build; cd build
cmake ..
make
export CUDA_HOME="/usr/local/cuda"
cd pytorch_binding
python setup.py install
```

Finally:

```
pip install -r requirements.txt
```

# Usage

## Dataset

Currently supports AN4, TEDLIUM and Voxforge.

### AN4

To download and setup the an4 dataset run below command in the root folder of the repo:

```
cd data; python an4.py
```

### TEDLIUM

You have the option to download the raw dataset file manually or through the script (which will cache it).
The file is found [here](http://www.openslr.org/resources/19/TEDLIUM_release2.tar.gz).

To download and setup the TEDLIUM_V2 dataset run below command in the root folder of the repo:

```
cd data; python ted.py # Optionally if you have downloaded the raw dataset file, pass --tar_path /path/to/TEDLIUM_release2.tar.gz

```
### Voxforge

To download and setup the Voxforge dataset run below command in the root folder of the repo:

```
cd data; python voxforge.py
```

Note that this dataset does not come with a validation dataset or test dataset.

### Custom Dataset

To create a custom dataset you must create a CSV file containing the locations of the training data. This has to be in the format of:

```
/path/to/audio.wav,/path/to/text.txt
/path/to/audio2.wav,/path/to/text2.txt
...
```

The first path is to the audio file, and the second path is to a text file containing the transcript on one line. This can then be used as stated below.

## Training

```
python train.py --train_manifest data/train_manifest.csv --val_manifest data/val_manifest.csv
```

Use `python train.py --help` for more parameters and options.

There is also [Visdom](https://github.com/facebookresearch/visdom) support to visualise training. Once a server has been started, to use:

```
python train.py --visdom
```

## Checkpoints

Training supports saving checkpoints of the model to continue training from should an error occur or early termination. To enable epoch
checkpoints use:

```
python train.py --checkpoint
```

To enable checkpoints every N batches through the epoch as well as epoch saving:

```
python train.py --checkpoint --checkpoint_per_batch N # N is the number of batches to wait till saving a checkpoint at this batch.
```

Note for the batch checkpointing system to work, you cannot change the batch size when loading a checkpointed model from it's original training
run.

To continue from a checkpointed model that has been saved:

```
python train.py --continue_from models/deepspeech_checkpoint_epoch_N_iter_N.pth.tar
```

This continues from the same training state as well as recreates the visdom graph to continue from if enabled.