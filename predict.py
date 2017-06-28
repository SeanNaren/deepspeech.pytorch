import argparse
import re
import numpy as np
import os
import soundfile as sf

import torch
from torch.autograd import Variable

from data.data_loader import SpectrogramParser
from decoder import ArgMaxDecoder
from model import DeepSpeech
from spell import correction

def np_softmax(tensor):
    arr = tensor.cpu().numpy() 
    exp = np.exp(arr-np.max(arr))
    return exp/exp.sum()

def finalize_ctm(ctm, seconds_per_timestep):
    chars = ''.join(ctm['chars'])

    if ctm['start_ts'] is None or ctm['end_ts'] is None:
        return None

    conf = np.mean(ctm['probs'])
    start_sec = seconds_per_timestep * ctm['start_ts']
    duration = ctm['end_ts'] * seconds_per_timestep - start_sec

    return {'chars': chars, 'word': '', 'start': float("{:.2f}".format(start_sec)), 'duration': float("{:.2f}".format(duration)), 'conf': float("{:.2f}".format(conf))}

def load_model(model_path, cuda=False):
    model = DeepSpeech.load_model(model_path, cuda=cuda)
    model.eval()

    labels = DeepSpeech.get_labels(model)
    audio_conf = DeepSpeech.get_audio_conf(model)

    decoder = ArgMaxDecoder(labels)
    parser = SpectrogramParser(audio_conf, normalize=True)

    return model, labels, audio_conf, decoder, parser

def predict(audio_path, model, labels, audio_conf, decoder, parser, debug=False, transcript_path=None):
    spect = parser.parse_audio(audio_path).contiguous()
    spect = spect.view(1, 1, spect.size(0), spect.size(1))
    out = model(Variable(spect, volatile=True))
    out = out.transpose(0, 1)  # TxNxH

    probs = out.data
    decoded_output = decoder.decode(probs)

    int_to_char = dict([(i, c) for (i, c) in enumerate(labels)])
    transposed = probs.transpose(0,1)
    _, max_probs = torch.max(transposed,2)
    softmax_probs = []

    f = sf.SoundFile(audio_path)
    duration = len(f)/float(f.samplerate)
    seconds_per_timestep = duration/transposed.size()[1]

    # loop through each timestep and store the largest softmax value of all the characters
    ctms = []
    ctm = {'chars': [], 'probs': [], 'start_ts': None, 'end_ts': None}
    last_char = ""
    chars = []
    for i in range(transposed.size()[1]):
        char = int_to_char[np.argmax(np_softmax(transposed[:,i]))]
        chars.append(char)
        if char == ' ':
            if not last_char == ' ':
                c = finalize_ctm(ctm, seconds_per_timestep)
                if not c is None:
                    ctms.append(c)
                ctm = {'chars': [], 'probs': [], 'start_ts': None, 'end_ts': None}
        else:
            if char != '_':
                ctm['chars'].append(char)
                ctm['probs'].append(np.max(np_softmax(transposed[:,i])))
                if ctm['start_ts'] is None:
                    ctm['start_ts'] = i
                ctm['end_ts'] = i

        last_char = char

    if len(ctm['chars']) > 0:
        c = finalize_ctm(ctm, seconds_per_timestep)
        if not c is None:
            ctms.append(c)

    # if there is a space before apostrophe, which happens frequently, it has to be joined with the previous ctm
    # so that the words match up with the language model output
    for i in range(len(ctms)):
        if i+1 < len(ctms) and ctms[i+1]['chars'][0] == "'":
            ctms[i]['chars'] = ctms[i]['chars'] + ctms[i+1]['chars']
            ctms[i]['duration'] = float("{:.2f}".format(ctms[i]['duration'] + ctms[i+1]['duration']))
            ctms[i]['conf'] = float("{:.2f}".format((ctms[i]['conf'] + ctms[i+1]['conf'])/2))
            del ctms[i+1]

    output = decoded_output[0]
    corrected = correction(output)

    corrected_words = corrected.split()
    for i in range(len(corrected_words)):
        if i < len(ctms):
            ctms[i]['word'] = corrected_words[i]

    #print([(catch['word'],catch['start']) for catch in ctms])
    if debug:
        print('')
        print("duration: {}s".format(duration))
        print('')
        print(ctms)
        print('')
        print(''.join(chars))
        print('')
        print(output)
        print('')

        if transcript_path != None and os.path.isfile(transcript_path):
            with open(transcript_path) as f:
                transcript = f.read()

            print("WER: {:.3f}%".format(decoder.wer(corrected, transcript)/float(len(transcript.split()))*100))

        print(corrected)

    return ctms

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepSpeech prediction')
    parser.add_argument('--model_path', default='models/latest.pth.tar',
                        help='Path to model file created by training')
    parser.add_argument('--audio_path', default='audio.wav',
                        help='Audio file to predict on')
    parser.add_argument('--transcript_path', default='transcript.txt',
                        help='Trascript file to validate on')
    parser.add_argument('--debug', action="store_true", help='print debug logs', default=False)
    parser.add_argument('--cuda', action="store_true", help='Use cuda to test model', default=False)
    args = parser.parse_args()

    model, labels, audio_conf, decoder, parser = load_model(args.model_path, args.cuda)
    ctms = predict(args.audio_path, model, labels, audio_conf, decoder, parser, debug=args.debug, transcript_path=args.transcript_path)
    if not args.debug:
        words = [c['word'] for c in ctms]
        print(' '.join(words))

