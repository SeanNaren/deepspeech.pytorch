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

Install pytorch audio:
```
sudo apt-get install sox libsox-dev libsox-fmt-all
git clone https://github.com/pytorch/audio.git
cd audio
python setup.py install
```

Finally:
```
pip install -r requirements.txt
```

# Usage

## Dataset

Currently supports AN4, TEDLIUM, Voxforge and LibriSpeech. Scripts will setup the dataset and create manifest files used in dataloading.

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

To download and setup the Voxforge dataset run the below command in the root folder of the repo:

```
cd data; python voxforge.py
```

Note that this dataset does not come with a validation dataset or test dataset.

### LibriSpeech

To download and setup the LibriSpeech dataset run the below command in the root folder of the repo:

```
cd data; python librispeech.py
```

You have the option to download the raw dataset files manually or through the script (which will cache them as well).
In order to do this you must create the following folder structure and put the corresponding tar files that you download from [here](http://www.openslr.org/12/).

```
cd data/
mkdir LibriSpeech/ # This can be anything as long as you specify the directory path as --target_dir when running the librispeech.py script
mkdir LibriSpeech/val/
mkdir LibriSpeech/test/
mkdir LibriSpeech/train/
```

Now put the `tar.gz` files in the correct folders. They will now be used in the data pre-processing for librispeech and be removed after
formatting the dataset.

Optionally you can specify the exact librispeech files you want if you don't want to add all of them. This can be done like below:

```
cd data/
python librispeech.py --files_to_use "train-clean-100.tar.gz, train-clean-360.tar.gz,train-other-500.tar.gz, dev-clean.tar.gz,dev-other.tar.gz, test-clean.tar.gz,test-other.tar.gz"
```

### Custom Dataset

To create a custom dataset you must create a CSV file containing the locations of the training data. This has to be in the format of:

```
/path/to/audio.wav,/path/to/text.txt
/path/to/audio2.wav,/path/to/text2.txt
...
```

The first path is to the audio file, and the second path is to a text file containing the transcript on one line. This can then be used as stated below.


### Merging multiple manifest files

To create bigger manifest files (to train/test on multiple datasets at once) we can merge manifest files together like below from a directory
containing all the manifests you want to merge. You can also prune short and long clips out of the new manifest.

```
cd data/
python merge_manifests.py --output_path merged_manifest.csv --merge_dir all_manifests/ --min_duration 1 --max_duration 15 # durations in seconds
```

## Training

```
python train.py --train_manifest data/train_manifest.csv --val_manifest data/val_manifest.csv
```

Use `python train.py --help` for more parameters and options.

There is also [Visdom](https://github.com/facebookresearch/visdom) support to visualise training. Once a server has been started, to use:

```
python train.py --visdom
```

### Noise Augmentation/Injection

There is support for two different types of noise; noise augmentation and noise injection.

#### Noise Augmentation

Applies small changes to the tempo and gain when loading audio to increase robustness. To use, use the `--augment` flag when training.

#### Noise Injection

Dynamically adds noise into the training data to increase robustness. To use, first fill a directory up with all the noise files you want to sample from.
The dataloader will randomly pick samples from this directory.

To enable noise injection, use the `--noise_dir /path/to/noise/dir/` to specify where your noise files are. There are a few noise parameters to tweak, such as
`--noise_prob` to determine the probability that noise is added, and the `--noise_min`, `--noise_max` parameters to determine the minimum and maximum noise to add in training.

Included is a script to inject noise into an audio file to hear what different noise levels/files would sound like. Useful for curating the noise dataset.

```
python noise_inject.py --input_path /path/to/input.wav --noise_path /path/to/noise.wav --output_path /path/to/input_injected.wav --noise_level 0.5 # higher levels means more noise
```

### Checkpoints

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

### Choosing batch sizes

Included is a script that can be used to benchmark whether training can occur on your hardware, and the limits on the size of the model/batch
sizes you can use. To use:

```
python benchmark.py --batch_size 32
```

Use the flag `--help` to see other parameters that can be used with the script.

### Model details

Saved models contain the metadata of their training process. To see the metadata run the below command:

```
python model.py --model_path models/deepspeech.pth.tar
```

To also note, there is no final softmax layer on the model as when trained, warp-ctc does this softmax internally. This will have to also be implemented in complex decoders if anything is built on top of the model, so take this into consideration!

## Testing/Inference

To evaluate a trained model on a test set (has to be in the same format as the training set):

```
python test.py --model_path models/deepspeech.pth.tar --test_manifest /path/to/test_manifest.csv --cuda
```

An example script to output a prediction has been provided:

```
python predict.py --model_path models/deepspeech.pth.tar --audio_path /path/to/audio.wav
```

## Acknowledgements

Thanks to [Egor](https://github.com/EgorLakomkin) and [Ryan](https://github.com/ryanleary) for their contributions!
