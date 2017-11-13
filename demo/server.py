import torch
import os
from tempfile import NamedTemporaryFile
import subprocess
from flask import Flask, request, jsonify
from torch.autograd import Variable
from data.data_loader import SpectrogramParser
from decoder import GreedyDecoder
from model import DeepSpeech


app = Flask(__name__)
ALLOWED_EXTENSIONS = set(['.wav', '.mp3', '.ogg', '.webm'])

speech_transcriber = None

class SpeechTranscriber:
    def __init__(self, model_path):
        """

        :param model_path:
        """
        assert os.path.exists(model_path), "Cannot find model here {}".format(model_path)
        self.deep_speech_model = DeepSpeech.load_model(model_path)
        self.deep_speech_model.eval()
        labels = DeepSpeech.get_labels(self.deep_speech_model)
        self.audio_conf = DeepSpeech.get_audio_conf(self.deep_speech_model)
        self.decoder = GreedyDecoder(labels)
        self.parser = SpectrogramParser(self.audio_conf, normalize=True)

    def transcribe(self, audio_file):
        """

        :param audio_file:
        :return:
        """
        spect = self.parser.parse_audio(audio_file).contiguous()
        spect = spect.view(1, 1, spect.size(0), spect.size(1))
        out = self.deep_speech_model(Variable(spect, volatile=True))
        out = out.transpose(0, 1)  # TxNxH
        decoded_output = self.decoder.decode(out.data)
        return decoded_output


@app.route('/transcribe', methods=['POST'])
def transcribe_file():
    """

    :return:
    """
    res = {}
    if request.method == 'POST':
        if 'file' not in request.files:
            res['status'] = "error"
            res['message'] = "audio file should be passed for the transcription"
            return jsonify(res)
        file = request.files['file']
        filename = file.filename
        _, file_extension = os.path.splitext(filename)
        if file_extension.lower() not in ALLOWED_EXTENSIONS:
            res['status'] = "error"
            res['message'] = "{} is not supported format.".format(file_extension)
            return jsonify(res)
        with NamedTemporaryFile(suffix=file_extension) as tmp_saved_audio_file:
            file.save( tmp_saved_audio_file.name )
            target_file = tmp_saved_audio_file.name.replace(file_extension, '_converted.wav')
            with open(os.devnull, 'w') as devnull:
                subprocess.call(["ffmpeg", '-i', tmp_saved_audio_file.name, "-ar", '16000', "-ab", "32", target_file],
                                stdout=devnull, stderr=devnull)
            transcription = speech_transcriber.transcribe(target_file)[0]
            res['status'] = "OK"
            res['transcription'] = transcription
            return jsonify(res)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='DeepSpeech transcription server')
    parser.add_argument('--model_path', default='./../models/deepspeech_final.pth.tar',
                        help='Path to model file created by training')
    parser.add_argument('--port', type=int, default=8888, help='Port to be used by the server')
    opt = parser.parse_args()

    global speech_transcriber
    speech_transcriber = SpeechTranscriber(model_path=opt.model_path)
    app.run(host='0.0.0.0',
            port=opt.port, debug=True, use_reloader=False,)

if __name__ == "__main__":
    main()