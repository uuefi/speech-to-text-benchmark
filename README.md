cache# Speech-to-Text Benchmark

This is a minimalist and extensible framework for benchmarking different speech-to-text engines. It has been developed
and tested on Ubuntu 20 (x86_64) using Python3.6.

## Table of Contents

* [Background](#background)
* [Data](#data)
* [Metrics](#metrics)
    * [Word Error Rate](#word-error-rate)
    * [Real Time Factor](#real-time-factor)
    * [Model Size](#model-size)
* [Usage](#Install)
    * [Word Error Rate Measurement](#word-error-rate-measurement)
    * [Real Time Factor Measurement](#real-time-factor-measurement)
* [Results](#results)
* [License](#license)

## Background

Automatic benchmark of several speech-to-text APIs and edge frameworks.

## Data

Cleaned data are downloaded from gdrive automatically.

## Metrics

This benchmark considers three metrics: word error rate (+mer, wil), real-time factor, and model size.

### Word Error Rate

Word error rate (WER) is defined as the ratio of [Levenstein distance](https://en.wikipedia.org/wiki/Levenshtein_distance)
between words in a reference transcript and words in the output of the speech-to-text engine, to the number of
words in the reference transcript.

### Real Time Factor

Real time factor (RTF) is measured as the ratio of CPU (processing) time to the length of the input speech file. A
speech-to-text engine with lower RTF is more computationally efficient. We omit this metric for cloud-based engines.

### Model Size

The aggregate size of models (acoustic and language), in MB. We omit this metric for cloud-based engines.

## online Speech-to-Text Engines

### Amazon Transcribe

Amazon Transcribe is a cloud-based speech recognition engine, offered by AWS. Find more information [here](https://aws.amazon.com/transcribe/).
1. Create a config file at ~/.aws/credentials with
```
[default]
aws_access_key_id=xxx
aws_secret_access_key=xxx
region=us-west-2
```
Alternatively, set the variables via AmazonCLI (aws configure).

### Google Speech-to-Text

A cloud-based speech recognition engine offered by Google Cloud Platform. Find more information
[here](https://cloud.google.com/speech-to-text/).
1. [Set up project, service account and store key] (https://cloud.google.com/speech-to-text/docs/before-you-begin)
2. export GOOGLE_APPLICATION_CREDENTIALS=".google/sunlit-adviser-325911-b113c509bf4c.json"

### Microsoft azure

## offline Speech-to-Text Engines

### Silero

[Download models](https://github.com/snakers4/silero-models)

# to-do: check if needed if pushed together
!pip install -q omegaconf torchaudio pydub
cd ./resources
git clone -q --depth 1 https://github.com/snakers4/silero-models
mv silero-models sileromodels

### Vosk

https://github.com/alphacep/vosk-api
[Download models](https://alphacephei.com/vosk/models)

### Picovoice Cheetah

[Cheetah](https://github.com/Picovoice/cheetah) is a streaming speech-to-text engine developed using
[Picovoice's](http://picovoice.ai/) proprietary deep learning technology. It works offline and is supported on a
growing number of platforms including Android, iOS, and Raspberry Pi.

### Picovoice Leopard

[Leopard](https://github.com/Picovoice/leopard) is a speech-to-text engine developed using
[Picovoice's](http://picovoice.ai/) proprietary deep learning technology. It works offline and is supported on a
growing number of platforms including Android, iOS, and Raspberry Pi.

### Mozilla DeepSpeech

Make sure that you have installed DeepSpeech on your machine by following the instructions on their official pages.
[Mozilla DeepSpeech](https://github.com/mozilla/DeepSpeech) is an open-source implementation of
[Baidu's DeepSpeech](https://arxiv.org/abs/1412.5567) by Mozilla.

### CMU PocketSphinx

Make sure that you have installed PocketSphinx on your machine by following the instructions on their official pages.
[PocketSphinx](https://github.com/cmusphinx/pocketsphinx) works offline and can run on embedded platforms such as
Raspberry Pi.

## Install

Below is information on how to use this framework to benchmark the speech-to-text engines. 

1. Download models and Unpack
e.g. DeepSpeech's models under [resources/deepspeech](/resources/deepspeech).
2. For running Google Speech-to-Text and Amazon Transcribe, you need to sign up for the respective cloud provider
and setup permissions / credentials according to their documentation. Running these services may incur fees.


## Examples
### Word Error Rate Measurement

Word Error Rate can be measured by running the following command from the root of the repository: 

```bash
python benchmark.py --engine_type AN_ENGINE_TYPE
```

The valid options for the `engine_type`
parameter are: `AMAZON_TRANSCRIBE`, `CMU_POCKET_SPHINX`, `GOOGLE_SPEECH_TO_TEXT`, `MOZILLA_DEEP_SPEECH`,
`PICOVOICE_CHEETAH`, `PICOVOICE_CHEETAH_LIBRISPEECH_LM`, `PICOVOICE_LEOPARD`, and `PICOVOICE_LEOPARD_LIBRISPEECH_LM`.

For testing, run the data generation with a small data sample:
```bash
python benchmark.py --test
```

### Real Time Factor Measurement

The `time` command is used to measure the execution time of different engines for a given audio file, and then divide
the CPU time by audio length. For example, to measure the execution time for DeepSpeech, run:

```bash
time deepspeech \
--model resources/deepspeech/output_graph.pbmm \
--lm resources/deepspeech/lm.binary \
--trie resources/deepspeech/trie \
--audio PATH_TO_WAV_FILE
```

The output should have the following format (values may be different):

```bash
real	0m4.961s
user	0m4.936s
sys	0m0.024s
```

Then, divide the `user` value by the length of the audio file, in seconds. The user value is the actual CPU time spent in the program.


## Results

The below results are obtained by following the previous steps. The benchmarking was performed on a Linux machine running
Ubuntu 20 with 
- Memory: 32GB of RAM 
- Compute: Intel® Core™ i7-10850H CPU @ 2.70GHz × 12
- Graphics NVIDIA Corporation TU106GLM Quadro RTX 3000 Mobile / Max-Q / Quadro RTX 3000/PCIe/SSE2

WER refers to word error rate and RTF refers to  real time factor.

1. CallHome
| Engine | WER | RTF (Desktop) | Model Size (Acoustic and Language) |
:---:|:---:|:---:|:---:|:---:|:---:
Amazon Transcribe | N/A| N/A | N/A | N/A | N/A |
CMU PocketSphinx (0.1.15) | N/A| N/A | N/A | N/A | N/A |
Google Speech-to-Text | N/A| N/A | N/A | N/A | N/A |
Mozilla DeepSpeech (0.6.1) | N/A | N/A | N/A | N/A | 1146.8 MB |
Picovoice Leopard (v1.0.0) | N/A | N/A | N/A | N/A | 47.9 MB |
Picovoice Leopard LibriSpeech LM (v1.0.0) | N/A | N/A | N/A | N/A | **45.0 MB** |
Silero (vx.0.0) | N/A | N/A | N/A | N/A | N/A |
Vosk (vx.0.0) | N/A | N/A | N/A | N/A | N/A |

2. Torsten
| Engine | WER | RTF (Desktop) | Model Size (Acoustic and Language) |
:---:|:---:|:---:|:---:|:---:|:---:
Amazon Transcribe | N/A| N/A | N/A | N/A | N/A |
CMU PocketSphinx (0.1.15) | N/A| N/A | N/A | N/A | N/A |
Google Speech-to-Text | N/A| N/A | N/A | N/A | N/A |
Mozilla DeepSpeech (0.6.1) | N/A | N/A | N/A | N/A | 1146.8 MB |
Picovoice Leopard (v1.0.0) | N/A | N/A | N/A | N/A | 47.9 MB |
Picovoice Leopard LibriSpeech LM (v1.0.0) | N/A | N/A | N/A | N/A | **45.0 MB** |
Silero (vx.0.0) | N/A | N/A | N/A | N/A | N/A |
Vosk (vx.0.0) | N/A | N/A | N/A | N/A | N/A |

3. Own recordings has no manual transcriptons yet

## License

The benchmarking framework is under the Apache 2.0 license.

Basic components are taken from Picovoice. The provided Cheetah and Leopard resources (binary, model, and license file) 
are the property of Picovoice. They are only to be used for evaluation purposes.

## To-dos
- testing
- clean up
- actual benchmark