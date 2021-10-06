#todo: refactor constants NAME_DATSET/ENGINE

import os
ROOT_PATH = os.path.dirname(__file__)

# DATA
DATA_PATH = os.path.join(ROOT_PATH, 'data')
NO_OF_FILES_FOR_TEST_MODE = 2


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
SAMPLE_RATE_AWS = 8000
IS_MONO_AWS = False #todo: check
IS_NORMALISE_AWS = False

# google
BUCKET_REGION = 'us'
LANGUAGE_GOOGLE = 'de-DE'
AUTH_GOOGLE_PATH = '/home/manu/.google/sunlit-adviser-325911-b113c509bf4c.json' # do not use ~/
SAMPLE_RATE_GOOGLE = 16000
IS_MONO_GOOGLE = True
IS_NORMALISE_GOOGLE = False

# CMUPocket
# todo: specify and test
MODEL_CMU_PATH = os.path.join(MODEL_PATH, 'resources/cmu', 'xxx')
cache
# GENERAL AUDIO
SAMPLE_RATE = None
CACHE_AUDIO = True



# EXPORT
EXPORT_PREDICTION_PATH = os.path.join(ROOT_PATH, 'export')

# CACHE
USE_CACHE = False
CACHE_PATH = os.path.join(ROOT_PATH, 'cache')