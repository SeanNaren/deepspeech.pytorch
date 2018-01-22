import os
import argparse
import wget
import subprocess
from utils import create_manifest
import tarfile

parser = argparse.ArgumentParser(description='Processes and downloads Mozilla Common Voice dataset.')
parser.add_argument("--target_dir", default='common_voice/', type=str, help="Directory to store the dataset.")
parser.add_argument("--output_dir", default='./', type=str, help="Directory to store the manifest files.")
parser.add_argument("--split_types", default='valid,other', type=str, help="Split types to process separated by comma."
                                                                           " valid, other and invalid are supported")
parser.add_argument('--sample_rate', default=16000, type=int, help='Sample rate')

args = parser.parse_args()

COMMON_VOICE_URL = "https://s3.us-east-2.amazonaws.com/common-voice-data-download/cv_corpus_v1.tar.gz"
DL_FILENAME = "cv_corpus_v1.tar.gz"


SPLITS_BY_TYPE = {
    "valid" : ["cv-valid-train", "cv-valid-dev", "cv-valid-test"],
    "invalid" :["cv-invalid"],
    "other" : ["cv-other-train", "cv-other-dev", "cv-other-test"]
}


def _preprocess_transcript(phrase):
    return phrase.strip().upper()

def read_transcriptions(transcriptions_file):
    transciptions = {}
    with open(transcriptions_file, "r") as f:
        next(f)
        for l in f:
            l = l.strip()
            if len(l) == 0:
                continue
            l = l.split(',')
            transciptions[l[0]] = _preprocess_transcript(l[1])
    return transciptions

def main():
    target_dl_dir = args.target_dir
    if not os.path.exists(target_dl_dir):
        os.makedirs(target_dl_dir)
    target_file = os.path.join(target_dl_dir, DL_FILENAME)
    if not os.path.exists(target_file):
        print("Downloading Mozilla Common Voice corpus...")
        wget.download(COMMON_VOICE_URL, target_dl_dir)
    split_types = args.split_types.lower().split(',')


    unpacked_dir = os.path.join(target_dl_dir, "cv_corpus_v1")
    if not os.path.exists(unpacked_dir):
        print("Unpacking {}...".format(target_file))
        tar = tarfile.open(target_file)
        tar.extractall(target_dl_dir)
        tar.close()

    for split_type in split_types:
        assert split_type in SPLITS_BY_TYPE, "{} split type is not supported. {} are supported".format(split_type,
                                                                                                SPLITS_BY_TYPE.keys())
        for split in SPLITS_BY_TYPE[split_type]:
            print("Processing {} split ...".format(split))
            split_dir = os.path.join(unpacked_dir, split)
            transcription_file = os.path.join(unpacked_dir,split + ".csv")

            wav_dir = os.path.join(split_dir, "wav")
            txt_dir = os.path.join(split_dir, "txt")
            os.makedirs(wav_dir, exist_ok=True)
            os.makedirs(txt_dir, exist_ok=True)

            transcriptions = read_transcriptions(transcription_file)
            for root, subdirs, files in os.walk(split_dir):
                for f in files:
                    if f.find(".mp3") != -1:
                        transcription = transcriptions[ split + '/' + f ]
                        wav_recording_path = os.path.join(wav_dir, f.replace(".mp3", ".wav"))
                        txt_path = os.path.join(txt_dir,  f.replace(".mp3", ".txt"))
                        if not os.path.exists(wav_recording_path):
                            subprocess.call(["sox \"{}\"  -r {} -b 16 -c 1 \"{}\"".format(os.path.join(split_dir, f), "16000",
                                                                            wav_recording_path)], shell=True)
                        if not os.path.exists(txt_path):
                            with open(txt_path, "w") as f:
                                f.write(transcription)
                                f.flush()
            print("Creating manifest {}".format(split))
            create_manifest(split_dir, os.path.join(args.output_dir, split.replace('-','_') + '_manifest.csv'))




if __name__ == "__main__":
    main()
