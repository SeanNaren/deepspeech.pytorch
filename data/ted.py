import os
import argparse
import subprocess
import unicodedata

from utils import create_manifest

parser = argparse.ArgumentParser(description='Processes and downloads TED-LIUMv2 dataset.')
parser.add_argument( "--target_dir", type = str, help = "Directory to store the dataset." )
parser.add_argument('--sample_rate', default=16000, type=int, help='Sample rate')
args = parser.parse_args()

TED_LIUM_V2_DL_URL = "http://www.openslr.org/resources/19/TEDLIUM_release2.tar.gz"


def get_utterances_from_stm(stm_file):
    """
    Return list of entries containing phrase and its start/end timings
    :param stm_file:
    :return:
    """
    res = []
    with open( stm_file, "r" ) as f:
        for stm_line in f:
            tokens = stm_line.split()
            start_time = float(tokens[3])
            end_time = float(tokens[4])
            filename = tokens[0]
            transcript = unicodedata.normalize("NFKD",
                                               " ".join(t for t in  tokens[6:]) ).\
                encode("ascii", "ignore").decode("ascii", "ignore")
            if transcript != "ignore_time_segment_in_scoring":
                res.append( {
                    "start_time" : start_time, "end_time" : end_time,
                    "filename" : filename, "transcript" : transcript
                } )
        return res

def cut_utterance(src_sph_file, target_wav_file, start_time, end_time, sample_rate = 16000):
    #subprocess.call( ["sox", "-r", str(sample_rate), "-b", "16", "-e",
    #                  "signed-integer", "-B", "-c", str(1),
    #                  src_sph_file, target_wav_file,
    #                  "trim", str(start_time), "={}".format( end_time )], shell= True )
    subprocess.call(["sox {}  -r {} -b 16 -c 1 {} trim {} ={}".format( src_sph_file, str(sample_rate),
                                                                                target_wav_file, start_time, end_time )], shell = True)

def _preprocess_transcript(phrase):
    return phrase.strip().upper()


def filter_short_utterances( utterance_info, min_len_sec = 1.0 ):
    return utterance_info["end_time"] -  utterance_info["start_time"] > min_len_sec

def prepare_dir( ted_dir ):
    converted_dir = os.path.join( ted_dir, "converted" )
    #directories to store converted wav files and their transcriptions
    wav_dir = os.path.join( converted_dir, "wav" )
    if not os.path.exists(wav_dir):
        os.makedirs( wav_dir )
    txt_dir = os.path.join( converted_dir, "txt" )
    if not os.path.exists(txt_dir):
        os.makedirs( txt_dir )

    for sph_file in os.listdir( os.path.join( ted_dir, "sph" ) ):
        speaker_name = sph_file.split('.sph')[0]

        sph_file_full = os.path.join( ted_dir, "sph", sph_file )
        stm_file_full = os.path.join( ted_dir, "stm", "{}.stm".format( speaker_name ) )

        print(stm_file_full, sph_file_full)
        assert os.path.exists( sph_file_full ) and os.path.exists( stm_file_full )
        all_utterances = get_utterances_from_stm(stm_file_full)

        all_utterances = filter( filter_short_utterances, all_utterances )
        for utterance_id, utterance in enumerate(all_utterances):
            target_wav_file = os.path.join( wav_dir, "{}_{}.wav".format( utterance["filename"], str(utterance_id) ) )
            target_txt_file =  os.path.join( txt_dir, "{}_{}.txt".format( utterance["filename"], str(utterance_id) ) )
            cut_utterance( sph_file_full, target_wav_file, utterance["start_time"], utterance["end_time"],
                           sample_rate = args.sample_rate )
            with open(target_txt_file, "w") as f:
                f.write( _preprocess_transcript(utterance["transcript"]) )
def main():
    target_dl_dir = args.target_dir
    if not os.path.exists( target_dl_dir ):
        os.makedirs( target_dl_dir )

    target_file =  os.path.join( target_dl_dir, "TEDLIUM_release2.tar.gz" )
    target_unpacked_dir = os.path.join( target_dl_dir, "TEDLIUM_release2" )
    if not os.path.exists( target_file ):
        print("Downloading corpus...")
        subprocess.call(['wget {} -P {}'.format( TED_LIUM_V2_DL_URL, target_dl_dir )], shell=True)
    if not os.path.exists( target_unpacked_dir ):
        print("Unpacking courpus...")
        os.makedirs( target_unpacked_dir )
        subprocess.call(["tar zxvf {} -C {}".format( target_file, target_dl_dir )], shell = True)

    train_ted_dir = os.path.join( target_unpacked_dir, "train" )
    val_ted_dir = os.path.join( target_unpacked_dir, "dev" )
    test_ted_dir = os.path.join(target_unpacked_dir, "test")

    prepare_dir( train_ted_dir )
    prepare_dir( val_ted_dir )
    prepare_dir( test_ted_dir )
    print('Creating manifests...')

    create_manifest(train_ted_dir, 'train')
    create_manifest(val_ted_dir, 'val')
    create_manifest(test_ted_dir, 'test')

if __name__ == "__main__":
    main()
