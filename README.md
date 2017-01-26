# deepspeech.pytorch

# TODO
* Create data-loader separate from aeon. This is due to dependency and just some strange implementations that could be handled with some more pre-processing
* WER/CER are not in line with what is expected
* Support LibriSpeech via multi-processed scripts
* Cleaner Warp-CTC bindings that does not rely on numpy

Implementation of [Baidu Warp-CTC](https://github.com/baidu-research/warp-ctc) using pytorch.
Creates a network based on the [DeepSpeech2](http://arxiv.org/pdf/1512.02595v1.pdf) architecture, trained with the CTC activation function.

# Installation

Several libraries are needed to be installed for training to work. I will assume that everything is being installed in
an Anaconda installation on Ubuntu.

Install pytorch if you haven't already:
```
conda install pytorch -c https://conda.anaconda.org/t/6N-MsQ4WZ7jo/soumith
```

Install the Nervana Aeon dataloader. Installation instructions can be seen [here](https://aeon.nervanasys.com/index.html/getting_started.html) Instructions to install for Anaconda/Ubuntu below:

```
sudo apt-get install libcurl4-openssl-dev clang libopencv-dev libsox-dev
git clone https://github.com/NervanaSystems/aeon.git
cd aeon
python setup.py install
```

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
pip install python-levenshtein
```

# Usage

## Dataset

Currently only supports an4. To download and setup the an4 dataset run below command in the root folder of the repo:

```
python get_an4.py
python create_dataset_manifest.py --root_path dataset/
```

This will generate csv manifests files used to load the data for training.

Optionally, a noise dataset can be used to inject noise artifically into the training data. Just fill a folder with noise wav files you want to inject (a source of noise files is the [musan dataset](http://www.openslr.org/17/)) and run the below command:
```
python create_noise_manifest.py --root_path noise/ # or whatever you've named your noise folder
```

## Training

You need to find the maximum duration of the training and testing samples. The command below will iterate through the current
folder and find the longest duration:

```
find . -type f -name "*.wav" | xargs soxi -D | sort | tail -n 1
```

Afterwards you can run the training script.

```
python main.py --max_duration 6.4 # This is the default max duration (for an4)
```

