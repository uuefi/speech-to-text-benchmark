#todo: refactor constants NAME_DATSET/ENGINE

import os
MA_PATH = '/Users/christina/Documents/02Masterarbeit/'
ROOT_PATH= '/Users/christina/Documents/02Masterarbeit/pycharm_2022-02-18/speech-to-text-benchmark/'

# DATA
DATA_PATH = os.path.join(MA_PATH, 'data')
NO_OF_FILES_FOR_TEST_MODE = 3

# torsten
MINIMUM_NO_WORDS_PER_FILE = 6
DATA_NAME_TORSTEN = 'torsten_de'
DATA_TORSTEN_PATH = os.path.join(DATA_PATH, DATA_NAME_TORSTEN)
DOWNLOAD_TORSTEN = '1oOMjNOK8oRPxu5ZVvog-9ZqPS85xYLPz'
SAMPLE_RATE_TORSTEN = 24000

# callhome
DATA_NAME_CALLHOME = 'callhome_de'
DATA_CALLHOME_PATH = os.path.join(DATA_PATH, DATA_NAME_CALLHOME)
DOWNLOAD_CALLHOME = '1fkP4fVZp5XL9dH2qi1-VORIbIxJDFFFY'
SAMPLE_RATE_CALLHOME = 8000

# forschergeist
DATA_NAME_FORSCHERGEIST = 'forschergeist_de'
DATA_FORSCHERGEIST_PATH = os.path.join(DATA_PATH, DATA_NAME_FORSCHERGEIST)
DOWNLOAD_FORSCHERGEIST = ''
SAMPLE_RATE_FORSCHERGEIST = 44100

# commonvoice
DATA_NAME_COMMONVOICE = 'commonvoice_de'
DATA_COMMONVOICE_PATH = os.path.join(DATA_PATH, DATA_NAME_COMMONVOICE)
DOWNLOAD_COMMONVOICE = '1fkP4fVZp5XL9dH2qi1-VORIbIxJDFFFY'
SAMPLE_RATE_COMMONVOICE = 48000

# verbmobil
DATA_NAME_VERBMOBIL = 'verbmobil_de'
DATA_VERBMOBIL_PATH = os.path.join(DATA_PATH, DATA_NAME_VERBMOBIL)
DOWNLOAD_VERBMOBIL = '1fkP4fVZp5XL9dH2qi1-VORIbIxJDFFFY'
SAMPLE_RATE_VERBMOBIL = 16000

# own
DATA_NAME_OWN = 'own_recordings'
DATA_OWN_PATH = os.path.join(DATA_PATH, DATA_NAME_OWN)
DOWNLOAD_OWN = '1oRJ3yjSpW-blWPH29stZ3HRDfZUy3K6u'
SAMPLE_RATE_OWN = 16000

# test
# todo: create a zip with one wave file + transcript for test runs
DATA_NAME_TEST = 'test'
DATA_TEST_PATH = os.path.join(DATA_PATH, DATA_NAME_TEST)
DOWNLOAD_TEST = '1oOMjNOK8oRPxu5ZVvog-9ZqPS85xYLPz'

# libri - english only
DATA_LIBRISPEECH_PATH = os.path.join(DATA_PATH, 'resources/data/LibriSpeech/test-clean')

# MODELS
MODEL_PATH = os.path.join(ROOT_PATH, 'resources')

# silero
MODEL_SILERO_PATH = os.path.join(MODEL_PATH, 'sileromodels')
SILERO_DEVICE = 'cpu'  # you can use any pytorch device
SILERO_MODELS = os.path.join(MODEL_SILERO_PATH, 'models.yml')
SILERO_USE_VAD = False
SAMPLE_RATE_SILERO = 16000
IS_MONO_SILERO = False # todo: check
IS_NORMALISE_SILERO = False # todo: check

# vosk
MODEL_VOSK_SMALL_PATH = os.path.join(MODEL_PATH, 'vosk/python/example', 'vosk-model-small-de-0.15')
MODEL_VOSK_PATH = os.path.join(MODEL_PATH, 'vosk/python/example', 'vosk-model-de-0.21')
BATCH_VOSK_DURATION = 4000
SAMPLE_RATE_VOSK = 16000
IS_MONO_VOSK = False # todo: check
IS_NORMALISE_VOSK = False # todo: check

# aws
LANGUAGE_AWS = 'de-DE' #'en-US'
REGION_AWS = 'us-west-2'
# todo: check optimal sample rate requirmenet from aws (8000)
SAMPLE_RATE_AWS = 16000
IS_MONO_AWS = False #todo: check
IS_NORMALISE_AWS = False

# azure
LANGUAGE_AZURE = 'de-DE' #'en-US'
REGION_AZURE = 'southcentralus'
KEY_AZURE = 'db085cc1f69b42eea62267921df68700'
STORAGE_CONNECTION_STRING_AZURE= 'DefaultEndpointsProtocol=https;AccountName=blobazure123;AccountKey=jUhLlDqxZBe6Ct8FLF2YvK42yATX2sm48fu2CKmdHpemoYT+c7VckJhWzCDYxrAEojzcaRKevNFZIslg4wfp1w==;EndpointSuffix=core.windows.net'
BLOB_URI_AZURE = 'https://blobazure123.blob.core.windows.net/firstcontainer?sp=r&st=2022-02-02T09:41:55Z&se=2022-02-28T17:41:55Z&sv=2020-08-04&sr=c&sig=2Gvr71pBCjvI9quYaVGi14RjkZHYM0UxXHmkE7Z6jEQ%3D'
SAS_URI_BLOB = 'https://blobazure123.blob.core.windows.net/?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2022-02-28T17:56:23Z&st=2022-02-02T09:56:23Z&spr=https&sig=575VKOqxl9URubFg7P4giMKCs6cAYRbAsdDFzJVaiPM%3D'
CONTAINER_NAME_AZURE = 'firstcontainer'
SAMPLE_RATE_AZURE = 16000
IS_MONO_AZURE = False #todo: check
IS_NORMALISE_AZURE = False

# google
BUCKET_REGION = 'us'
BUCKET_NAME_GOOGLE = 'asr_bucket123'
LANGUAGE_GOOGLE = 'de-DE'
AUTH_GOOGLE_PATH = '/Users/christina/Documents/02Masterarbeit/Google/pristine-ally-339114-2fe038f3b473.json' # do not use ~/
SAMPLE_RATE_GOOGLE = 16000
IS_MONO_GOOGLE = True
IS_NORMALISE_GOOGLE = False

# CMUPocket
# todo: specify and test
MODEL_CMU_PATH = os.path.join(MODEL_PATH, 'resources/cmu', 'xxx')
#cache
# GENERAL AUDIO
SAMPLE_RATE = None
CACHE_AUDIO = True

# Picovoice

LANGUAGE_PICOVOICE = 'de-DE'
AUTH_PICOVOICE_PATH = 'IiotzO/opdMMsL3PasTvLgbs3jF2ntkRjfgxCnbtteHhGNG9tSQsYw==' # do not use ~/
SAMPLE_RATE_PICOVOICE = 16000
IS_MONO_PICOVOICE = True
IS_NORMALISE_PICOVOICE = False

# EXPORT
EXPORT_PREDICTION_PATH = os.path.join(MA_PATH, 'export')

# CACHE
USE_CACHE = False
CACHE_PATH = os.path.join(MA_PATH, 'cache')