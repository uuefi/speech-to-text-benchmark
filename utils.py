import os
import pandas as pd
import json
import config
from download_data import download_file_from_google_drive


def check_data_available(dataset_path):
    if not os.path.exists(dataset_path):
        return False
    else:
        return True


def percentage_wrapper(value):
    return round(value * 100, 2)


def download_data(dataset_path: str, download_id: str):
    print("Data {dataset_path} does not exist. Downloading...")
    download_file_from_google_drive(dataset_path, download_id, destination=config.DATA_PATH)


def transcript_name(wav_path):
    if os.path.basename(wav_path).split('.')[-1] == 'flac':
        return wav_path.split(os.path.sep)[-1].replace(".flac", ".txt")
    if os.path.basename(wav_path).split('.')[-1] == 'wav':
        return wav_path.split(os.path.sep)[-1].replace(".wav", ".txt")

def transcript_json_name(wav_path):
    if os.path.basename(wav_path).split('.')[-1] == 'flac':
        return wav_path.split(os.path.sep)[-1].replace(".flac", ".json")
    if os.path.basename(wav_path).split('.')[-1] == 'wav':
        return wav_path.split(os.path.sep)[-1].replace(".wav", ".json")


def blob_name(wav_path):
    path = wav_path.split(os.path.sep)[-2]
    name = wav_path.split(os.path.sep)[-1]  # .replace(".wav", ".txt")

    return os.path.join(path, name)
