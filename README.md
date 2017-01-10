# deepspeech.pytorch

# Installation

Several libraries are needed to be installed for training to work. I will assume that everything is being installed in
an Anaconda installation on Ubuntu.

Follow instructions [here](https://aeon.nervanasys.com/index.html/getting_started.html) to install Nervanas' dataloader aeon. Instructions for Anaconda/Ubuntu below:

```
sudo apt-get install libcurl4-openssl-dev clang libopencv-dev libsox-dev
git clone https://github.com/NervanaSystems/aeon.git
cd aeon
python setup.py install
```
Install pytorch if you haven't already:

```
conda install pytorch -c https://conda.anaconda.org/t/6N-MsQ4WZ7jo/soumith
```

Install this fork for Warp-CTC bindings:

```
git clone https://github.com/SeanNaren/ctc.git
cd ctc
mkdir build
cd build
cmake ..
make
sudo /home/<your.name>/anaconda2/bin/python setup.py install # e.g sudo /home/sean.narenthiran/anaconda2/bin/python setup.py install
```

# Usage

## Dataset

Currently only supports an4. To download and setup the an4 dataset run below command in the root folder of the repo:

```
python get_an4.py
```
