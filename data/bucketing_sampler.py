from torch.utils.data.sampler import Sampler
import numpy as np
from data.data_loader import SpectrogramDataset, load_audio
from collections import defaultdict


class SpectrogramDatasetWithLength(SpectrogramDataset):
    def __init__(self, *args, **kwargs):
        """
        SpectrogramDataset that splits utterances into buckets based on their length.
        Bucketing is done via numpy's histogram method.
        Used by BucketingSampler to sample utterances from the same bin.
        """
        super(SpectrogramDatasetWithLength, self).__init__(*args, **kwargs)
        audio_paths = [path for (path, _) in self.ids]
        audio_lengths = [len(load_audio(path)) for path in audio_paths]
        hist, bin_edges = np.histogram(audio_lengths, bins="auto")
        audio_samples_indices = np.digitize(audio_lengths, bins=bin_edges)
        self.bins_to_samples = defaultdict(list)
        for idx, bin_id in enumerate(audio_samples_indices):
            self.bins_to_samples[bin_id].append(idx)


class BucketingSampler(Sampler):
    def __init__(self, data_source):
        """
        Samples from a dataset that has been bucketed into bins of similar sized sequences to reduce
        memory overhead.
        :param data_source: The dataset to be sampled from
        """
        super(BucketingSampler, self).__init__(data_source)
        self.data_source = data_source
        assert hasattr(self.data_source, 'bins_to_samples')

    def __iter__(self):
        for bin, sample_idx in self.data_source.bins_to_samples.items():
            np.random.shuffle(sample_idx)
            for s in sample_idx:
                yield s

    def __len__(self):
        return len(self.data_source)
