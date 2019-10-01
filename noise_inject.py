import argparse

import torch
from scipy.io.wavfile import write

from data.data_loader import load_audio, NoiseInjection

parser = argparse.ArgumentParser()
parser.add_argument('--input-path', default='input.wav', help='The input audio to inject noise into')
parser.add_argument('--noise-path', default='noise.wav', help='The noise file to mix in')
parser.add_argument('--output-path', default='output.wav', help='The noise file to mix in')
parser.add_argument('--sample-rate', default=16000, help='Sample rate to save output as')
parser.add_argument('--noise-level', type=float, default=1.0,
                    help='The Signal to Noise ratio (higher means more noise)')
args = parser.parse_args()

noise_injector = NoiseInjection()
data = load_audio(args.input_path)
mixed_data = noise_injector.inject_noise_sample(data, args.noise_path, args.noise_level)
mixed_data = torch.tensor(mixed_data, dtype=torch.float).unsqueeze(1)  # Add channels dim
write(filename=args.output_path,
      data=mixed_data.numpy(),
      rate=args.sample_rate)
print('Saved mixed file to %s' % args.output_path)
