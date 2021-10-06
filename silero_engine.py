# imports for uploading/recording
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.io import wavfile
from torchaudio.functional import vad

from config import SILERO_DEVICE, SILERO_USE_VAD, SILERO_MODELS, SAMPLE_RATE
from resources.sileromodels.utils import (init_jit_model,
                                          split_into_batches,
                                          read_batch,
                                          prepare_model_input)

use_VAD = SILERO_USE_VAD


def get_model():
    return OmegaConf.load(SILERO_MODELS)


def get_device():
    return torch.device(SILERO_DEVICE)  # you can use any pytorch device


def apply_vad(audio, boot_time=0, trigger_level=9, **kwargs):
    print('\nVAD applied\n')
    vad_kwargs = dict(locals().copy(), **kwargs)
    vad_kwargs['sample_rate'] = SAMPLE_RATE
    del vad_kwargs['kwargs'], vad_kwargs['audio']
    audio = vad(torch.flip(audio, ([0])), **vad_kwargs)
    return vad(torch.flip(audio, ([0])), **vad_kwargs)


# wav to text method
def wav_to_text(f='./test.wav'):
    batch = read_batch([f])
    model_input = prepare_model_input(batch, device=get_device())
    model_output = model(model_input)
    return decoder(model_output[0].cpu())


def wav_to_text(f='./test.wav'):
    batch = read_batch([f])
    model_input = prepare_model_input(batch, device=get_device())
    model_output = model(model_input)
    return decoder(model_output[0].cpu())


def recognize(audio):
    if use_VAD:
        audio = apply_vad(audio)
    wavfile.write('./test.wav', SAMPLE_RATE, (32767 * audio).astype(np.int16))
    transcription = wav_to_text()
    print(transcription)


if __name__ == '__main__':
    # todo: create test case
    model, decoder = init_jit_model(models.stt_models.de.latest.jit, device=get_device())
    batches = split_into_batches(test_files, batch_size=10)
    model_input = prepare_model_input(read_batch(batches[0]),
                                      device=get_device())

    assert isinstance(model_input, object)
    output = model(model_input)
    for example in output:
        print(decoder(example.cpu()))
