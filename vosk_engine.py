from vosk import Model, KaldiRecognizer, SetLogLevel
import json
import subprocess
from config import BATCH_VOSK_DURATION, SAMPLE_RATE


def load_model(model_path):
    model = Model(model_path)
    rec = KaldiRecognizer(model, SAMPLE_RATE)
    rec.SetMaxAlternatives(1)
    rec.SetWords(True)
    SetLogLevel(0)
    return rec


def transform_and_transcribe(rec, file):
    process = subprocess.Popen(['ffmpeg', '-loglevel', 'quiet', '-i',
                                file,
                                '-ar', str(SAMPLE_RATE), '-ac', '1', '-f', 's16le', '-'],
                               stdout=subprocess.PIPE)

    text = []
    list_of_json = []

    while True:
        data = process.stdout.read(BATCH_VOSK_DURATION)
        if len(data) == 0:
            break

        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            # todo: develop a systematic that when more than one alternatives are used not just the last one is used
            list_of_json.append(res['alternatives'][-1]['result'])
            text.append(res['alternatives'][-1]['text'])

        flat_list = [item for sublist in list_of_json for item in sublist]
    return " ".join(text), flat_list


if __name__ == '__main__':
    # todo: small test
    print('create a new test .wav')
