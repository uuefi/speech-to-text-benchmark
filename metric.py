import editdistance
import jiwer
from jiwer import wer, mer, wil

from utils import transcript_name, percentage_wrapper


def init_metric():
    return {
        'transcript_name': [],
        'ref_transcript_word_count': [],
        'pred_transcript_word_count': [],
        'word_error_count': [],
        'wer': [],
        'mer': [],
        'wil': []
    }


def calculate_metrics(metric, path, true_words, pred_words):
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.RemoveKaldiNonWords(),  # <unk>, [laugh] etc.
        jiwer.RemoveWhiteSpace(replace_by_space=False),  # whitespace characters are , \t, \n, \r, \x0b and \x0c
        jiwer.SentencesToListOfWords(word_delimiter=" ")
    ])

    metric['transcript_name'].append(transcript_name(path))
    metric['word_error_count'].append(editdistance.eval(true_words, pred_words))

    metric['wer'].append(percentage_wrapper(wer(true_words,
                                                pred_words,
                                                truth_transform=transformation,
                                                hypothesis_transform=transformation
                                                )))
    metric['mer'].append(percentage_wrapper(mer(true_words,
                                                pred_words,
                                                hypothesis_transform=transformation,
                                                truth_transform=transformation
                                                )))
    metric['wil'].append(percentage_wrapper(wil(true_words,
                                                pred_words,
                                                truth_transform=transformation,
                                                hypothesis_transform=transformation
                                                )))
    metric['ref_transcript_word_count'].append(len(true_words))
    metric['pred_transcript_word_count'].append(len(pred_words))

    return metric
