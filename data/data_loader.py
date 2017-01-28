import scipy.signal
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import librosa
import numpy as np

windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}


class AudioDataset(Dataset):
    def __init__(self, audio_conf, manifest_filepath, alphabet, normalize=False):
        """
        Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
        a comma. Each new line is a different sample. Example below:

        /path/to/audio.wav,/path/to/audio.txt
        ...

        :param audio_conf: Dictionary containing the sample rate, window type and the window length/stride in seconds
        :param manifest_filepath: Path to manifest csv as describe above
        :param alphabet: String containing all the possible characters to map to
        :param normalize: Apply standard mean and deviation normalization to audio tensor
        """
        super(AudioDataset, self).__init__()
        with open(manifest_filepath) as f:
            ids = f.readlines()
        ids = [x.strip().split(',') for x in ids]
        self.ids = ids
        self.size = len(ids)
        self.audio_conf = audio_conf
        self.window = windows.get(audio_conf['window'], windows['hamming'])
        self.alphabet_map = dict([(alphabet[i], i) for i in range(len(alphabet))])
        self.normalize = normalize

    def __getitem__(self, index):
        sample = self.ids[index]
        audio_path, transcript_path = sample[0], sample[1]
        spect = self._spectrogram(audio_path)
        transcript = self._parse_transcript(transcript_path)
        return spect, transcript

    def _parse_transcript(self, transcript_path):
        with open(transcript_path, 'r') as transcript_file:
            transcript = transcript_file.read().replace('\n', '')
        transcript = [self.alphabet_map[x] for x in list(transcript)]
        return transcript

    def _spectrogram(self, audio_path):
        y, _ = librosa.core.load(audio_path, sr=self.audio_conf['sample_rate'])
        n_fft = int(self.audio_conf['sample_rate'] * self.audio_conf['window_size'])
        win_length = n_fft
        hop_length = int(self.audio_conf['sample_rate'] * self.audio_conf['window_stride'])
        # STFT
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=self.window)
        spect, phase = librosa.magphase(D)
        # S = log(S+1)
        spect = np.log1p(spect)
        spect = torch.FloatTensor(spect)
        if self.normalize:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)

        return spect

    def __len__(self):
        return self.size


def _collate_fn(batch):
    def func(p):
        return p[0].size(1)

    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
        target_sizes[x] = len(target)
        targets.extend(target)
    targets = torch.IntTensor(targets)
    return inputs, targets, input_percentages, target_sizes


class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn
