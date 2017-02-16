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

Currently only supports an4. To download and setup the an4 dataset run below command in the root folder of the repo:

```
cd data; python an4.py
```

This will generate csv manifests files used to load the data for training.

LibriSpeech formatting is in the works.

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

