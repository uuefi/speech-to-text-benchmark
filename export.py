import os
import json
import pandas as pd

from config import EXPORT_PREDICTION_PATH
from utils import transcript_name, transcript_json_name


def export_path(date, params):
    return os.path.join(EXPORT_PREDICTION_PATH,
                        f"{params.dataset_name}_{params.engine_type}_{params.test}_{date}")


def export_transcript(transcript, path, date, params):
    filename = transcript_name(path)
    path = os.path.join(export_path(date, params), 'txt')
    os.makedirs(path, exist_ok=True)
    print(f"export transcript: {os.path.join(path, filename)}")
    with open(os.path.join(path, filename), "w") as text_file:
        text_file.write(transcript)


def export_timestamp_transcript(transcript, path, date, params):
    filename = transcript_json_name(path)
    path = os.path.join(export_path(date, params), 'json')
    os.makedirs(path, exist_ok=True)
    print(f"export transcript: {os.path.join(path, filename)}")
    with open(os.path.join(path, filename), "w") as text_file:
        json.dump(transcript, text_file)


def export_metric(metric, date, params):
    path = export_path(date, params)
    pd.DataFrame.from_dict(metric).to_csv(os.path.join(path, 'summary.csv'), index=False)
