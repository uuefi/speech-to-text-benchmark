import argparse
from datetime import datetime

import metric
from dataset import *
from engine import *
from export import export_transcript, export_timestamp_transcript, export_metric
from processing import transform_audio
from utils import transcript_name
import jiwer

# import normalise as normalise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine_type', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--test', dest='test', action='store_true')
    parser.add_argument('--cache', dest='cache', action='store_true')
    parser.add_argument('--export', dest='export_transcript', action='store_true')
    args = parser.parse_args()

    date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")

    dataset = Dataset.create(args.dataset_name, args.test)
    print('loaded %s with %.2f hours of data' % (str(dataset), dataset.size_hours()))

    engine = ASREngine.create(ASREngines[args.engine_type],args.dataset_name)
    print('created %s engine' % str(engine))

    metrics = metric.init_metric()

    for i in range(dataset.size()):
        path, ref_transcript = dataset.get(i)
        print (path)
        # sample rate and mono is engine depending, normalise data
        path = transform_audio(args, path, engine.sample_rate, engine.is_mono, engine.is_normalised)

        transcript, timestamp_transcript = engine.transcribe(path)

        ref_words = ref_transcript.split()
        words = transcript.split()

        print(f'Create transcript for {transcript_name(path)}')
        if args.export_transcript:
            # todo: speaker diarisation
            export_transcript(transcript, path, date, args)
            export_timestamp_transcript(timestamp_transcript, path, date, args)

        if dataset.HAS_TRANSCRIPT:
            metrics = metric.calculate_metrics(metrics, path, ref_words, words)

    if dataset.HAS_TRANSCRIPT:
        print('word error rate : %.2f' % (
                100 * float(sum(metrics['word_error_count'])) / sum(metrics['ref_transcript_word_count'])))
        print('word error rate : %.2f' % (sum(metrics['wer']) / len(metrics['wer'])))
        export_metric(metrics, date, args)
