import os
import wave
from warnings import warn

from pydub import AudioSegment
from pydub.utils import mediainfo
import librosa
import numpy as np
import scipy
import soundfile as sf


# helper to figure out the issue
def frame_rate_channel(audio_file_name, head):
    #x, _ = librosa.load(audio_file_name, sr=16000)
    #sf.write(os.path.join(head,'tmp.wav'), x, 16000)

    with wave.open(os.path.join(head,'tmp.wav'), "rb") as wave_file:
        frame_rate = wave_file.getframerate()
        channels = wave_file.getnchannels()
        print(f'frame_rate: {frame_rate}')
        print(f'channels: {channels}')
        return frame_rate, channels


def transform_audio(args, path, sample_rate, to_mono, normalise=False):
    # todo: find a better way to hold the cache path here --> args
    path_target = path.replace('data', 'cache')#.replace('.wav', '.flac')
    head, tail = os.path.split(path_target)
    os.makedirs(head, exist_ok=True)

    # to be removed: helper output to figure out the issue
    info = mediainfo(path)
    print(info)
    print(info['sample_rate'])
    print(info['channels'])

    # remove one pipeline here: helper output to figure out the issue
    tranform_with_librosa(path_target, path, sample_rate, to_mono, normalise)
    #tranform_with_pydub(path_target, path, sample_rate, to_mono, normalise, info)

    # to be removed: helper output to figure out the issue
    #audiofile.export(path_target, format="flac")
    # scipy.io.wavfile.write(path_target, sample_rate, audio)
    # sf.write(path_target, audio, sample_rate)
    info = mediainfo(path_target)
    print(info)
    frame_rate_channel(path_target, head)

    return path_target


# experiment with to pipelines to figure what the issue is
def tranform_with_librosa(path_target, path, sample_rate, to_mono, normalise):
    audio, sr = librosa.core.load(path, sr=None)
    if int(sr) > sample_rate:
        print(f" - Sample rate can be reduced from {sr} to {sample_rate} for file {path}")
        audio = librosa.resample(audio, sr, sample_rate)
    elif int(sr) < sample_rate:
        warn(f" x Sample rate too low  from {sr} to {sample_rate} for file {path} (speech-to-text). "
             f"Upsample for technical reasons, however, effect is limited.")
        audio = librosa.resample(audio, sr, sample_rate)

    if to_mono:
        print(" - to mono")
        audio = librosa.to_mono(audio)

    if normalise:
        print(" - normalise")
        audio = audio * (0.7079 / np.max(np.abs(audio)))
        maxv = np.iinfo(np.int16).max
        audio = (audio * maxv).astype(np.int16)

    # librosa.output.write_wav(path_target, audio, sample_rate)
    # transfrom from 64-bit RIFF to flac
    sf.write(path_target, audio, sample_rate)



def tranform_with_pydub(path_target, path, sample_rate, to_mono, normalise, info):
    audiofile = AudioSegment.from_file(path)
    if int(info['sample_rate']) > sample_rate:
        print(f" - Sample rate can be reduced from {info['sample_rate']} to {sample_rate} for file {path}")
        audiofile = audiofile.set_frame_rate(sample_rate)
    elif int(info['sample_rate']) < sample_rate:
        warn(f" x Sample rate too low  from {info['sample_rate']} to {sample_rate} for file {path} (speech-to-text). "
             f"Upsample for technical reasons, however, effect is limited.")
        audiofile = audiofile.set_frame_rate(sample_rate)

    if to_mono:
        print(" - to mono")
        audiofile = audiofile.set_channels(1)

    if normalise:
        exit("Not implemented yet")

    audiofile.export(path_target, format="WAV")
