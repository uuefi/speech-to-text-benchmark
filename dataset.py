import os
import soundfile
import glob
import json
import pandas as pd
from abc import abstractmethod, ABC

import config
import download_data
from download_data import download_file_from_google_drive, check_data_available
from config import DATA_LIBRISPEECH_PATH, DATA_CALLHOME_PATH, DOWNLOAD_CALLHOME, DATA_TORSTEN_PATH, DATA_OWN_PATH, \
    DOWNLOAD_OWN, DOWNLOAD_TORSTEN, DATA_PATH, MINIMUM_NO_WORDS_PER_FILE, NO_OF_FILES_FOR_TEST_MODE


class Dataset(object):
    # todo: video vs. audio handle
    @abstractmethod
    def size(self):
        raise NotImplementedError()

    def size_hours(self):
        return sum(soundfile.read(self.get(i)[0])[0].size / (16000 * 3600) for i in range(self.size()))

    @abstractmethod
    def get(self, index):
        raise NotImplementedError()

    @abstractmethod
    def __str__(self):
        raise NotImplementedError()

    @classmethod
    def create(cls, dataset_type, sample):
        if dataset_type == 'librispeech':
            # todo sample
            return LibriSpeechDataset()
        elif dataset_type == config.DATA_NAME_CALLHOME:
            # todo: check sample rates for each dataset or transform!
            config.SAMPLE_RATE = config.SAMPLE_RATE_CALLHOME
            return CallHomeDataset(sample)
        elif dataset_type == config.DATA_NAME_TORSTEN:
            config.SAMPLE_RATE = config.SAMPLE_RATE_TORSTEN
            return TorstenDataset(sample)
        elif dataset_type == config.DATA_NAME_OWN:
            config.SAMPLE_RATE = config.SAMPLE_RATE_OWN
            return OwnRecordingsDataset(sample)
        else:
            raise ValueError("cannot create %s of type '%s'" % (cls.__name__, dataset_type))


class LibriSpeechDataset(Dataset, ABC):
    def __init__(self):
        self._data = list()
        self.HAS_TRANSCRIPT = True

        for speaker_id in os.listdir(DATA_LIBRISPEECH_PATH):
            speaker_dir = os.path.join(DATA_LIBRISPEECH_PATH, speaker_id)

            for chapter_id in os.listdir(speaker_dir):
                chapter_dir = os.path.join(speaker_dir, chapter_id)

                transcript_path = os.path.join(chapter_dir, '%s-%s.trans.txt' % (speaker_id, chapter_id))
                with open(transcript_path, 'r') as f:
                    transcripts = dict(x.split(' ', maxsplit=1) for x in f.readlines())

                for flac_file in os.listdir(chapter_dir):
                    if flac_file.endswith('.flac'):
                        wav_file = flac_file.replace('.flac', '.wav')
                        wav_path = os.path.join(chapter_dir, wav_file)
                        if not os.path.exists(wav_path):
                            flac_path = os.path.join(chapter_dir, flac_file)
                            pcm, sample_rate = soundfile.read(flac_path)
                            soundfile.write(wav_path, pcm, sample_rate)

                        self._data.append((wav_path, transcripts[wav_file.replace('.wav', '')]))

    def size(self):
        return len(self._data)

    def get(self, index):
        return self._data[index]

    def __str__(self):
        return 'LibriSpeech'


class CallHomeDataset(Dataset):

    def __init__(self, sample):
        self._data = list()
        self.HAS_TRANSCRIPT = True

        if not check_data_available(DATA_CALLHOME_PATH):
            download_file_from_google_drive(DATA_CALLHOME_PATH, DOWNLOAD_CALLHOME)

        audio_path = os.path.join(DATA_CALLHOME_PATH, 'audio_snippets/')
        transcript_path = os.path.join(DATA_CALLHOME_PATH, 'json/')

        file_no = len(glob.glob(transcript_path + '*.json'))
        assert file_no > 0, f'No transcripts, check data path {transcript_path}*.json'

        # limit the amount of available files to test mode
        if sample:
            print(f"Test mode! Number of files reduced to {NO_OF_FILES_FOR_TEST_MODE}!")
            file_no = NO_OF_FILES_FOR_TEST_MODE

        for transcript_file in glob.glob(transcript_path + '*.json')[:file_no]:
            with open(transcript_file, 'r') as f:
                data = json.load(f)
            wav_file = os.path.join(audio_path,
                                    transcript_file.split(os.path.sep)[-1].replace('.json', '.wav'))
            self._data.append((wav_file, data['transcripts']))

    def size(self):
        return len(self._data)

    def get(self, index):
        return self._data[index]

    def __str__(self):
        return config.DATA_NAME_CALLHOME


class TorstenDataset(Dataset, ABC):
    # Transcription service ask for a minimum fee per sample. To avoid cost, only examples with a mimimum length are
    # choosen
    MINIMUM_NO_WORDS_PER_FILE = 6

    def __init__(self, sample):
        self._data = list()
        self.HAS_TRANSCRIPT = True

        if not check_data_available(DATA_TORSTEN_PATH):
            download_file_from_google_drive(DATA_TORSTEN_PATH, DOWNLOAD_TORSTEN)

        audio_path = os.path.join(DATA_TORSTEN_PATH, 'wavs/')
        transcript_path = os.path.join(DATA_TORSTEN_PATH, 'metadata.csv')

        transcript_df = pd.read_csv(os.path.join(transcript_path), sep='|', dtype=str, names=["filename", "transcript"])
        transcript_df = transcript_df[transcript_df['transcript'].str.split().str.len().gt(MINIMUM_NO_WORDS_PER_FILE)]

        file_no = len(transcript_df.index)
        # len(glob.glob(transcript_path + '*.json'))
        # assert file_no > 0, f'No transcripts, check data path {transcript_path}*.json'

        # limit the amount of available files to test mode
        if sample:
            print(f"Test mode! Number of files reduced to {NO_OF_FILES_FOR_TEST_MODE}!")
            file_no = NO_OF_FILES_FOR_TEST_MODE

        for index, row in transcript_df[:file_no].iterrows():
            wav_file = os.path.join(audio_path, row['filename'] + '.wav')
            self._data.append((wav_file, row['transcript']))

    def size(self):
        return len(self._data)

    def get(self, index):
        return self._data[index]

    def __str__(self):
        return config.DATA_NAME_TORSTEN


class OwnRecordingsDataset(Dataset):
    def __init__(self, sample):
        self._data = list()
        self.HAS_TRANSCRIPT = False

        if not check_data_available(DATA_OWN_PATH):
            download_data.download_file_from_google_drive(DATA_OWN_PATH, DOWNLOAD_OWN)

        audio_path = os.path.join(DATA_OWN_PATH, 'audio/')

        audio_files = sorted(glob.glob(audio_path + '*.wav'), key=os.path.getsize)
        file_no = len(audio_files)
        # limit the amount of available files to test mode
        if sample:
            print(f"Test mode! Number of files reduced to {NO_OF_FILES_FOR_TEST_MODE}!")
            file_no = 1  # smallest audio file only

        for wav_file in audio_files[:file_no]:
            self._data.append((wav_file, None))

    def size(self):
        return len(self._data)

    def get(self, index):
        return self._data[index]

    def __str__(self):
        return config.DATA_NAME_OWN
