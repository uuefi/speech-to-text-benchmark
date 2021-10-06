import json
import os
import string
import subprocess
import time
import uuid
from enum import Enum

import boto3
import numpy as np
import requests
import soundfile
from deepspeech import Model
# from google.cloud import speech
from google.cloud import speech
from google.cloud import storage
# to be removed: helper output to figure out the issue
#from google.cloud import speech1b1 as speech
# from google.cloud.speech_v1 import enums
# from google.cloud.speech_v1 import types
# from pocketsphinx import get_model_path
# from pocketsphinx.pocketsphinx import Decoder
import config
from cache import write_cache, get_cache
from resources.sileromodels.utils import (init_jit_model,
                                          split_into_batches,
                                          read_batch,
                                          prepare_model_input)
from silero_engine import get_model, get_device
from utils import blob_name
from vosk_engine import load_model, transform_and_transcribe


class ASREngines(Enum):
    # online
    AMAZON_TRANSCRIBE = "AMAZON_TRANSCRIBE"
    GOOGLE_SPEECH_TO_TEXT = "GOOGLE_SPEECH_TO_TEXT"

    # edge
    CMU_POCKET_SPHINX = 'CMU_POCKET_SPHINX'
    MOZILLA_DEEP_SPEECH = 'MOZILLA_DEEP_SPEECH'
    PICOVOICE_CHEETAH = "PICOVOICE_CHEETAH"
    PICOVOICE_CHEETAH_LIBRISPEECH_LM = "PICOVOICE_CHEETAH_LIBRISPEECH_LM"
    PICOVOICE_LEOPARD = "PICOVOICE_LEOPARD"
    PICOVOICE_LEOPARD_LIBRISPEECH_LM = "PICOVOICE_LEOPARD_LIBRISPEECH_LM"
    SILERO = "SILERO"
    VOSK = "VOSK"


class ASREngine(object):
    def transcribe(self, path):
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()

    @classmethod
    def create(cls, engine_type, dataset_name):
        if engine_type is ASREngines.AMAZON_TRANSCRIBE:
            return AmazonTranscribe()
        elif engine_type is ASREngines.CMU_POCKET_SPHINX:
            return CMUPocketSphinxASREngine()
        elif engine_type is ASREngines.GOOGLE_SPEECH_TO_TEXT:
            return GoogleSpeechToText(dataset_name)
        elif engine_type is ASREngines.MOZILLA_DEEP_SPEECH:
            return MozillaDeepSpeechASREngine()
        elif engine_type is ASREngines.PICOVOICE_CHEETAH:
            return PicovoiceCheetahASREngine()
        elif engine_type is ASREngines.PICOVOICE_CHEETAH_LIBRISPEECH_LM:
            return PicovoiceCheetahASREngine(lm='language_model_librispeech.pv')
        elif engine_type is ASREngines.PICOVOICE_LEOPARD:
            return PicovoiceLeopardASREngine()
        elif engine_type is ASREngines.PICOVOICE_LEOPARD_LIBRISPEECH_LM:
            return PicovoiceLeopardASREngine(lm='language_model_librispeech.pv')
        elif engine_type is ASREngines.SILERO:
            return SILEROASREngine()
        elif engine_type is ASREngines.VOSK:
            return VOSKASREngine(config.MODEL_VOSK_PATH)

        else:
            raise ValueError("cannot create %s of type '%s'" % (cls.__name__, engine_type))


class AmazonTranscribe(ASREngine):
    def __init__(self):
        self.sample_rate = config.SAMPLE_RATE_AWS
        self.is_mono = config.IS_MONO_AWS
        self.is_normalised = config.IS_NORMALISE_AWS

        self._s3 = boto3.client('s3', region_name=config.REGION_AWS)
        self._s3_bucket = str(uuid.uuid4())
        self._s3.create_bucket(
            ACL='private',
            Bucket=self._s3_bucket,
            CreateBucketConfiguration={'LocationConstraint': config.REGION_AWS})
        self._transcribe = boto3.client('transcribe')

    def transcribe(self, path):

        # todo: cache solution
        if config.USE_CACHE:
            res, list_of_timestamp_words = get_cache(path)
        else:
            res, list_of_timestamp_words = None, None

        if res is None or list_of_timestamp_words is None:

            job_name = str(uuid.uuid4())
            s3_object = os.path.basename(path)
            self._s3.upload_file(path, self._s3_bucket, s3_object)

            self._transcribe.start_transcription_job(
                TranscriptionJobName=job_name,
                Media={'MediaFileUri': 'https://s3-us-west-2.amazonaws.com/%s/%s' % (self._s3_bucket, s3_object)},
                MediaFormat='wav',
                LanguageCode=config.LANGUAGE_AWS)

            while True:
                status = self._transcribe.get_transcription_job(TranscriptionJobName=job_name)
                if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
                    break
                time.sleep(5)

            content = requests.get(status['TranscriptionJob']['Transcript']['TranscriptFileUri'])

            full = json.loads(content.content.decode('utf8'))
            list_of_timestamp_words = []
            for item in full['results']['items']:
                list_of_timestamp_words.append(
                    dict(start_ts=item['start_time'], end_ts=item['end_time'], word=item['alternatives']['content']))

            res = json.loads(content.content.decode('utf8'))['results']['transcripts'][0]['transcript']
            res = res.translate(str.maketrans('', '', string.punctuation))

            write_cache(res, list_of_timestamp_words)

        return res, list_of_timestamp_words

    def __str__(self):
        return 'AMAZON TRANSCRIBE'


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )


class GoogleSpeechToText(ASREngine):
    #todo: fix this fucking bullshit
    def __init__(self, bucket_name):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.AUTH_GOOGLE_PATH
        self.sample_rate = config.SAMPLE_RATE_GOOGLE
        self.is_mono = config.IS_MONO_GOOGLE
        self.is_normalised = config.IS_NORMALISE_GOOGLE

        self.bucket_name = bucket_name
        # self.bucket = self.create_bucket_class_location()

        self._client = speech.SpeechClient()

    @property
    def create_bucket_class_location(self):
        """Create a new bucket in specific location with storage class"""
        storage_client = storage.Client()

        bucket = storage_client.bucket(self.bucket_name)
        bucket.storage_class = "COLDLINE"
        new_bucket = storage_client.create_bucket(bucket, location=config.BUCKET_REGION)

        print(
            "Created bucket {} in {} with storage class {}".format(
                new_bucket.name, new_bucket.location, new_bucket.storage_class
            )
        )
        return new_bucket

    # todo: add clean up
    def delete_blob(bucket_name, blob_name):
        """Deletes a blob from the bucket."""
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.delete()

    def transcribe(self, path):
        # example https://cloud.google.com/speech-to-text/docs/samples/speech-transcribe-async-gcs

        # with open(path, 'rb') as f:
        #    content = f.read()
        # client = speech.SpeechClient()7

        # akzeptiert auch keine manuell gepolished Daten via ffmpeg -i test.wav -acodec pcm_s16le -ac 1 -ar 16000 out.wav oder
        path = '/home/manu/Downloads/bg.wav'
            # '/home/manu/Desktop/Lukas/00_recoro/benchmark/speech-to-text-benchmark/cache/callhome_de/audio_snippets/6312.wav'
        upload_blob(self.bucket_name, path, blob_name(path))
        gcs_uri = f'gs://{self.bucket_name}/{blob_name(path)}'
        print(gcs_uri)
        audio = speech.RecognitionAudio(uri=gcs_uri)

        # self.create_bucket_class_location(self.bucket_name)
        # Detects speech in the audio file

        # todo: experiment with model microphone etc. can be specified e.g. phone_call, video
        # https://github.com/googleapis/python-speech/blob/a0bac07de9f4e89c41d34a47d9e35ec6fd7edac3/samples/snippets/beta_snippets.py
        #         sample_rate_hertz=8000,
        #         language_code="en-US",
        #         use_enhanced=True,
        #         # A model must be specified to use enhanced model.
        #         model="phone_call",

        # tested:
        # a) audio_channel_count = 2,
        # enable_separate_recognition_per_channel = True,
        encoding = speech.RecognitionConfig.AudioEncoding.FLAC
        # b) #speech.RecognitionConfig.AudioEncoding.LINEAR16,
        config_google = speech.RecognitionConfig(
            encoding=encoding,
            sample_rate_hertz=self.sample_rate,
            language_code=config.LANGUAGE_GOOGLE)

        operation = self._client.long_running_recognize(config=config_google, audio=audio)
        print("Waiting for operation to complete...")
        response = operation.result(timeout=900)

        for result in response.results:
            print(u"Transcript: {}".format(result.alternatives[0].transcript))
        # response = self._client.recognize(config_google, audio)

        res = ' '.join(result.alternatives[0].transcript for result in response.results)
        # res = res.translate(str.maketrans('', '', string.punctuation))

        #todo: json with timestamps

        return res

    def __str__(self):
        return 'Google Speech-to-Text'


class CMUPocketSphinxASREngine(ASREngine):
    # todo: add timestamp version
    def __init__(self):
        # todo: import/ install CMU package
        # https://github.com/cmusphinx/pocketsphinx-python/blob/master/example.py
        config = Decoder.default_config()
        config.set_string('-logfn', '/dev/null')
        config.set_string('-hmm', os.path.join(config.MODEL_CMU_PATH, 'en-us'))
        config.set_string('-lm', os.path.join(config.MODEL_CMU_PATH, 'en-us.lm.bin'))
        config.set_string('-dict', os.path.join(config.MODEL_CMU_PATH, 'cmudict-en-us.dict'))

        self._decoder = Decoder(config)

    def transcribe(self, path):
        pcm, sample_rate = soundfile.read(path)
        assert sample_rate == 16000
        pcm = (np.iinfo(np.int16).max * pcm).astype(np.int16).tobytes()

        self._decoder.start_utt()
        self._decoder.process_raw(pcm, no_search=False, full_utt=True)
        self._decoder.end_utt()

        words = []
        for seg in self._decoder.seg():
            word = seg.word

            # Remove special tokens.
            if word == '<sil>' or word == '<s>' or word == '</s>':
                continue

            word = ''.join([x for x in word if x.isalpha()])

            words.append(word)

        return ' '.join(words)

    def __str__(self):
        return 'CMUPocketSphinx'


class MozillaDeepSpeechASREngine(ASREngine):
    # todo: add timestamp version
    def __init__(self):
        # todo: set data processing
        deepspeech_dir = os.path.join(os.path.dirname(__file__), 'resources/deepspeech')
        model_path = os.path.join(deepspeech_dir, 'output_graph.pbmm')
        # alphabet_path = os.path.join(deepspeech_dir, 'alphabet.txt')
        language_model_path = os.path.join(deepspeech_dir, 'lm.binary')
        trie_path = os.path.join(deepspeech_dir, 'trie')

        # https://github.com/mozilla/DeepSpeech/blob/master/native_client/python/client.py
        self._model = Model(model_path, 500)
        self._model.enableDecoderWithLM(language_model_path, trie_path, 0.75, 1.85)

    def transcribe(self, path):
        pcm, sample_rate = soundfile.read(path)
        pcm = (np.iinfo(np.int16).max * pcm).astype(np.int16)
        res = self._model.stt(pcm)

        return res

    def __str__(self):
        return 'Mozilla DeepSpeech'


class PicovoiceCheetahASREngine(ASREngine):
    # todo: add timestamp version
    def __init__(self, lm='language_model.pv'):
        cheetah_dir = os.path.join(os.path.dirname(__file__), 'resources/cheetah')
        self._cheetah_demo_path = os.path.join(cheetah_dir, 'cheetah_demo')
        self._cheetah_library_path = os.path.join(cheetah_dir, 'libpv_cheetah.so')
        self._cheetah_acoustic_model_path = os.path.join(cheetah_dir, 'acoustic_model.pv')
        self._cheetah_language_model_path = os.path.join(cheetah_dir, lm)
        self._cheetah_license_path = os.path.join(cheetah_dir, 'cheetah_eval_linux.lic')

    def transcribe(self, path):
        args = [
            self._cheetah_demo_path,
            self._cheetah_library_path,
            self._cheetah_acoustic_model_path,
            self._cheetah_language_model_path,
            self._cheetah_license_path,
            path]
        res = subprocess.run(args, stdout=subprocess.PIPE).stdout.decode('utf-8')

        # Remove license notice
        filtered_res = [x for x in res.split('\n') if '[' not in x]
        filtered_res = '\n'.join(filtered_res)

        return filtered_res.strip('\n ')

    def __str__(self):
        return 'Picovoice Cheetah'


class PicovoiceLeopardASREngine(ASREngine):
    def __init__(self, lm='language_model.pv'):
        leopard_dir = os.path.join(os.path.dirname(__file__), 'resources/leopard')
        self._demo_path = os.path.join(leopard_dir, 'leopard_demo')
        self._library_path = os.path.join(leopard_dir, 'libpv_leopard.so')
        self._acoustic_model_path = os.path.join(leopard_dir, 'acoustic_model.pv')
        self._language_model_path = os.path.join(leopard_dir, lm)
        self._license_path = os.path.join(leopard_dir, 'leopard_eval_linux.lic')

    def transcribe(self, path):
        args = [
            self._demo_path,
            self._library_path,
            self._acoustic_model_path,
            self._language_model_path,
            self._license_path,
            path]
        res = subprocess.run(args, stdout=subprocess.PIPE).stdout.decode('utf-8')

        # Remove license notice
        filtered_res = [x for x in res.split('\n') if '[' not in x]
        filtered_res = '\n'.join(filtered_res)

        return filtered_res.strip('\n ')

    def __str__(self):
        return 'Picovoice Leopard'


class SILEROASREngine(ASREngine):

    def __init__(self):
        self.sample_rate = config.SAMPLE_RATE_SILERO
        self.is_mono = config.IS_MONO_SILERO
        self.is_normalised = config.IS_NORMALISE_SILERO

        self.models = get_model()
        self._device = get_device()
        self._model, self._decoder = init_jit_model(self.models.stt_models.de.latest.jit, device=self._device)

    def transcribe(self, path):
        # todo: batch 10 files at once and not file by file
        batches = split_into_batches([path], batch_size=10)

        audios = prepare_model_input(read_batch(batches[0]),
                                     device=self._device)

        # approximate timestamps linearly
        wav_len = audios.shape[1] / 16000

        output = self._model(audios)
        # words = [self._decoder(example.cpu()) for example in output]

        # {'word': 'ganz', 'start_ts': 0.44, 'end_ts': 0.64}
        list_of_timesamp_words = \
            [self._decoder(example.cpu(), wav_len, word_align=True)[-1] for i, example in enumerate(output)][0]
        words = [example['word'] for example in list_of_timesamp_words]

        res = " ".join(words)

        return res, list_of_timesamp_words

    def __str__(self):
        return 'Silero'


class VOSKASREngine(ASREngine):

    def __init__(self, vosk_path):
        self.sample_rate = config.SAMPLE_RATE_VOSK
        self.is_mono = config.IS_MONO_VOSK
        self.is_normalised = config.IS_NORMALISE_VOSK
        # todo: check if there are any pre-assumptions if the model init is only done once
        self._model = load_model(vosk_path)

    def transcribe(self, path):
        res, list_of_timesamp_words = transform_and_transcribe(self._model, path)

        return res, list_of_timesamp_words

    def __str__(self):
        return 'Vosk'
