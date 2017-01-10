import torch
from aeon import DataLoader
from neon.backends import gen_backend
import numpy as np
from torch.autograd import Variable

from CTCLoss import ctc_loss
from model import DeepSpeech

'''
Game plan:

Two parts, the data loading, and the actual training of the neural network.

Data loading: Once we have this, it should be scalable to the entire dataset.

Assume data is in the format of dataset/wav, dataset/txt, where the name is sample.{wav,txt}. Create manifest files by doing find on all wav
then assuming the transcript is in a folder called txt.

Using pytorch, we need to replicate the deepspeech architecture here. We should be able to pass a tensor through it, and get predictions out of it.

Next we need to use the python bindings for warp-ctc to train on it correctly.

Data has to be processed to 16 bit like this find . -name '*.wav'  | parallel 'sox {} -b 16 {} channels 1 rate 16k'
'''
sample_rate = 16000
home_path = '/home/sean/Work/deepspeech.pytorch/'
minibatch_size = 3
alphabet = "_'ABCDEFGHIJKLMNOPQRSTUVWXYZ "
nout = len(alphabet)
max_transcript_length = 1300
frame_length = .02
frame_stride = .01
spect_size = (frame_length * sample_rate / 2) + 1
be = gen_backend(batch_size=minibatch_size)

audio_config = dict(sample_freq_hz=sample_rate,
                    max_duration="7 seconds",
                    frame_length="%f seconds" % frame_length,
                    frame_stride="%f seconds" % frame_stride,
                    window_type='hamming',
                    noise_index_file="%smanifest_noise.csv" % home_path,
                    add_noise_probability=0.5,
                    noise_level=(0.5, 1.0)
                    )

transcription_config = dict(alphabet=alphabet,
                            max_length=max_transcript_length,
                            pack_for_ctc=True)

dataloader_config = dict(type="audio,transcription",
                         audio=audio_config,
                         transcription=transcription_config,
                         manifest_filename="%smanifest_train.csv" % home_path,
                         macrobatch_size=be.bsz,
                         minibatch_size=be.bsz)

train = DataLoader(dataloader_config, be)
data = train.next()
input = data[0].reshape(minibatch_size, 1, spect_size,
                        -1)  # Puts the data into the form of batch x channels x freq x time
input = Variable(torch.FloatTensor(input.get().astype(dtype=np.float32)))
target = Variable(torch.FloatTensor(data[1].get().astype(dtype=np.float32)))
label_lengths = torch.FloatTensor(data[2].get().astype(dtype=np.float32))

rnn_hidden_size = 200
batch_size = 1
model = DeepSpeech(rnn_hidden_size=rnn_hidden_size)
hidden = Variable(torch.randn(2, batch_size, rnn_hidden_size))
cell = Variable(torch.randn(2, batch_size, rnn_hidden_size))
print(model)
model = model.cuda()
input = input.cuda()
hidden = hidden.cuda()
cell = cell.cuda()

out = model(input, hidden, cell)

print(out.size())

sizes = Variable(torch.FloatTensor(out.size(1)).fill_(out.size(0)), requires_grad=False) # TODO we could probably use
#   the valid percentage to find out the real size

loss = ctc_loss(out, target, sizes, label_lengths)
grads = loss.backward()
