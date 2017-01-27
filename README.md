# deepspeech.pytorch

* Fix WER/CER measurements
* Add tests for dataloading
* Script to download an4 and create manifests (streamline rather than 2 separate scripts)
* Support LibriSpeech via multi-processed scripts

Implementation of [Baidu Warp-CTC](https://github.com/baidu-research/warp-ctc) using pytorch.
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
python get_an4.py
python create_dataset_manifest.py --root_path dataset/
```

This will generate csv manifests files used to load the data for training.

## Training


```
python main.py --train_manifest train_manifest.csv --test_manifest test_manifest.csv
```

