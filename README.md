# Subtitle2go - Open Source Automatic Subtitling

Subtitle2go is a fully automatic solution for German video subtitling, currently with a focus on lecture videos. The project uses open source models and scripts for German ASR, automatic punctuation reconstruction and subtitle segmentation. But it should be straight-forward to support other languages as well and PRs are welcome!

Subtitle2go recently added support for multiple ASR engines:

## Whisper

[Whisper](https://github.com/openai/whisper) produces high quality transcriptions (if you use the 'large' model family), many languages are supported, but it's very slow on CPUs. A GPU system is recommended. Note: you need our [Lecture2Go fork of Whisper](https://github.com/lecture2go/whisper).

## Speechcatcher

[Speechcatcher](https://github.com/speechcatcher-asr/speechcatcher) is much faster than Whisper, recommended for CPU only systems. Speechcatcher models are trained on Whisper output (teacher/student training). Allows for custom segmentation with our segmentation algorithm, you can make longer or shorter subtitle segments depending on your preferences. Only German is currently supported.

Note: you currently need the [timestamps branch version](https://github.com/speechcatcher-asr/speechcatcher/tree/timestamps) of Speechcatcher.

## Kaldi

[Kaldi](https://kaldi-asr.org/) based models are the fastest models for CPU, but won't receive updates anymore and are less accurate than Speechcatcher or Whisper models, espacially with noisy recordings. As it's not an end-to-end model, the output does not include punctuation. We use punctuation reconstruction to add it in a post processing step.

Our German Kaldi models are based on the [kaldi-tuda-de](https://github.com/uhh-lt/kaldi-tuda-de) TDNN-HMM recipe.

[rpunct](https://github.com/Felflare/rpunct) is used for punctuation reconstruction (,.?!)

Subtitle2go uses a custom solution for segmentation, with a beam search segmentation algorithm that searches for the best "segmentation path" with user specified criteria such as average length and variance. Punctuation and distances in parsing trees are used to estimate splitting costs.

![Subtitle2Go Kaldi Pipeline](Pipeline.png)

## News

### 13.12.2023

+ Small bug fixes and improved support for the Speechcatcher ASR engine.

### 08.12.2023

+ First release of version 2.0, supporting multiple ASR engines: Kaldi, Speechcatcher and Whisper.

### 02.09.2022

+ Lots of improvements to the German ASR model, punctuation reconstruction (we now use rpunct instead of punctuator2) and a preliminary English ASR model were added. See our new [paper](https://aclanthology.org/2022.konvens-1.11/) for more details.

+ Robert Geislinger is presenting the improvements at [Konvens2022](https://konvens2022.uni-potsdam.de/?page_id=65)!

### 01.11.2021

+ https://lecture2go.uni-hamburg.de/open-source is using subtitle2go in its production system!

### 03.03.2021
+ We presented Subtitle2Go at [ESSV2021](http://www.essv.de/essv2021/program/). You can check out our paper [here](http://www.essv.de/essv2021/pdfs/33_milde_v2.pdf) and [here (mirror)](https://www.inf.uni-hamburg.de/en/inst/ab/lt/publications/2021-mildeetal-subtitle.pdf).

### 24.10.2020
+ Added punctuation model (with punctuator2)
+ Added a better subtitle segmentation algorithm

## Requirements
+ Python 3.9 (recommended), or Python 3.8/3.7
+ Ubuntu 20.04 (other versions may work as well)
+ X86_64 processor

## Installation guide

This installation guide shows you how to install all dependencies, scripts and how to downloads pre-trained models (Kaldi+punctuator2).

```
# Ubuntu: make sure you have Python 3.11, its dev package and ffmpeg installed:
sudo apt-get update
sudo apt-get install python3.11 python3.11-dev ffmpeg

# Ubuntu: if python3.11 is not available, you can activate the deadsnakes Python PPA, in order to install python 3.11 through this PPA
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.11 python3.11-dev

# Now clone the subtitle2go package somwhere:
mkdir ~/projects/
cd ~/projects/
git clone https://github.com/uhh-lt/subtitle2go
cd subtitle2go/

# create virtual env and install python dependencies:

virtualenv -p /usr/bin/python3.11 subtitle2go_env
source subtitle2go_env/bin/activate
# You may need to check https://pytorch.org/ for PyTorch installation instruction
# E.g. If you need to install PyTorch with a different CUDA version
pip3 install numpy torch torchaudio 
pip3 install -r requirements.txt
python3 -m spacy download de_core_news_lg
python3 -m spacy download en_core_web_lg

# You can now check if all dependencies are correctly installed by calling the check_installation.py script.
# Note that it should tell you that it can't find Kaldi, but everything else should show 'success' after the checks.
# If you don't plan on using Kaldi models, this is fine!
python3 check_installation.py

######################################################################################
#
# The steps below are only needed if you would like to run the legacy Kaldi models.
# Speechcatcher and Whisper download and cache their ASR models at runtime.
#
######################################################################################

pip3 install -r requirements_kaldi.txt

# Install Kaldi and Intel MKL (see note below if you have a different CPU than Intel)

./install_mkl.sh
./install_kaldi_intel.sh ~/projects/subtitle2go/subtitle2go_env/bin/python3

# OR if you have a non-Intel CPU:
./install_kaldi.sh ~/projects/subtitle2go/subtitle2go_env/bin/python3

# Download and extract models
./download_models.sh

# you can now check if all dependencies are correctly installed by calling the check_installation.py script:
python3 check_installation.py

```

## Optional: redis status updates

subtitle2go.py can optionally send status updates to a redis instance. You can either connect to the redis server channel "subtitle2go" directly and receive update events. Alternatively you can run event_server.py to get a HTTP API to poll the status of all past and current subtitle2go.py runs.

The following packages only need to be installed if you want to use this feature. Subtitle2go.py will automatically figure out if the redis package is available and enables/disables the accordingly.

```
sudo apt-get redis-server
```

and on Mac Os X:

```
brew install redis
```

## Subtitle2go.py usage

Put a mediafile (eg `mediafile.mp4`) in the directory and then run:

```
source subtitle2go_env/bin/activate
. path.sh
python3 subtitle2go.py mediafile.mp4
```

The subtitle is then generated as `mediafile.vtt`

Optional: If you want to send status updates of the processing to redis/event_server then you need to append --with-redis-updates to subtitle2go.py:

```
python3 subtitle2go.py --with-redis-updates mediafile.mp4
```

If you start the wsgi server with ./start_eventserver_wsgi.sh then you can see status updates of every subtitle2go.py run at http://127.0.0.1:7500/status 

You can also clear successful and failed runs with http://127.0.0.1:7500/clear 

Redis server needs to be running.

# Subtitle2go.py program arguments

The following arguments are available:

```
usage: subtitle2go.py [-h] [-e {speechcatcher,kaldi,whisper}] [-s {vtt,srt}] [-l LANGUAGE] [-m MODEL_YAML] [-i ID] [-c CALLBACK_URL] [-p NUM_PROCS] [-o SUBTITLE_OFFSET]
                      [--rnn-rescore] [--acoustic-scale ACOUSTIC_SCALE] [--asr-beam-size ASR_BEAM_SIZE] [--asr-max-active ASR_MAX_ACTIVE]
                      [--segment-beam-size SEGMENT_BEAM_SIZE] [--ideal-token-len IDEAL_TOKEN_LEN] [--len-reward-factor LEN_REWARD_FACTOR]
                      [--sentence-end-reward_factor SENTENCE_END_REWARD_FACTOR] [--comma-end-reward-factor COMMA_END_REWARD_FACTOR]
                      [--whisper-task {transcribe,translate}] [--no-condition-on-previous-text] [--whisper-initial-prompt WHISPER_INITIAL_PROMPT]
                      [--whisper-no-speech-threshold WHISPER_NO_SPEECH_THRESHOLD] [--with-redis-updates] [--debug]
                      filename

positional arguments:
  filename              The path of the mediafile

options:
  -h, --help            show this help message and exit
  -e {speechcatcher,kaldi,whisper}, --engine {speechcatcher,kaldi,whisper}
                        The ASR engine to use. One of: kaldi, whisper or speechcatcher.
  -s {vtt,srt}, --subtitle {vtt,srt}
                        The output subtitleformat (vtt or srt). Default=vtt
  -l LANGUAGE, --language LANGUAGE
                        Sets the language of the models (de/en/...).With the engine option set to "whisper" you can also use "auto" for automatic language detection.
  -m MODEL_YAML, --model-yaml MODEL_YAML
                        Model used for decoding (yaml config for kaldi or model name for other engines).
  -i ID, --id ID        Manually sets the file id
  -c CALLBACK_URL, --callback-url CALLBACK_URL
                        Sets a callback URL to notify when process is finished or something went off
  -p NUM_PROCS, --num-procs NUM_PROCS
                        Number of parallel processors (Speechcatcher and Whisper only)
  -o SUBTITLE_OFFSET, --subtitle-offset SUBTITLE_OFFSET
                        Subtitle offset (in seconds)
  --rnn-rescore         Do RNNLM rescoring of the decoder output (only for PyKaldimodels).
  --acoustic-scale ACOUSTIC_SCALE
                        ASR decoder option: This is a scale on the acoustic log-probabilities, and is a universally used kludge in HMM-GMM and HMM-DNN systems to
                        account for the correlation between frames.
  --asr-beam-size ASR_BEAM_SIZE
                        ASR decoder option: controls the beam size in the beam search. This is a speed / accuracy tradeoff.
  --asr-max-active ASR_MAX_ACTIVE
                        ASR decoder option: controls the maximum number of states that can be active at one time.
  --segment-beam-size SEGMENT_BEAM_SIZE
                        What beam size to use for the segmentation search
  --ideal-token-len IDEAL_TOKEN_LEN
                        The ideal length of tokens per segment
  --len-reward-factor LEN_REWARD_FACTOR
                        How important it is to be close to ideal_token_len, higher factor = splits are closer to ideal_token_len
  --sentence-end-reward_factor SENTENCE_END_REWARD_FACTOR
                        The weight of the sentence end score in the search. Higher values make it more likely to always split at sentence end.
  --comma-end-reward-factor COMMA_END_REWARD_FACTOR
                        The weight of the comma end score in the search. Higher values make it more likely to always split at commas.
  --whisper-task {transcribe,translate}
                        The whisper task: one of either "transcribe" or "translate".
  --no-condition-on-previous-text
                        Disabling condition-on-previous-text will reduce Whispers accuracy, but can sometimes help to avoid hallucinations.
  --whisper-initial-prompt WHISPER_INITIAL_PROMPT
                        Initial prompt for the first segment. Can be used to pass inuseful additional information like an author name, a title, a custom vocabulary etc.
  --whisper-no-speech-threshold WHISPER_NO_SPEECH_THRESHOLD
                        Threshold parameter to decide if a segment is speechor not speech. Default is 0.6.
  --with-redis-updates  Update a redis instance about the current progress.
  --debug               Output debug timing information
```

## FAQ

### Error indicating a problem with "libcudnn_ops_infer.so.8"

It might be neccessary to reinstall torch with an older version of CUDA:

```
pip3 install --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

(Dec 2023)

### Can I upgrade the engines individually? 

Yes! As some dependencies are directly installed from git, it might be necessary to do a force reinstall:

For Speechcatcher:

```
pip3 install --force-reinstall --no-deps git+https://github.com/speechcatcher-asr/speechcatcher@timestamps
```

For Speechcatcher's espnet_streaming_decoder:

```
pip3 install --force-reinstall --no-deps git+https://github.com/speechcatcher-asr/espnet_streaming_decoder
```

For Whisper:

```
pip3 install --force-reinstall --no-deps git+https://github.com/lecture2go/whisper
```

This would only install the engine module by itself. You may need to leave out the --no-deps, if the dependnencies need to be upgraded too.

### What is Whisper's task parameter? 

Whisper supports two modes of operation. The default is 'transcribe' and the second mode is 'translate' (any-to-English). See this example, where the the video is in German:

```
$ python3 subtitle2go.py -e whisper --whisper-task "translate" subtitle2go-test.mp4
Using ASR beamsize: 5
Using Whisper as ASR engine.
self.filename='subtitle2go-test.mp4' self.fn_short_hash='17a23f553078047a' status='Starting Whisper decode.'
[00:01.000 --> 00:04.000]  Hello and welcome to Lecture to go.
[00:05.000 --> 00:07.000]  Are you looking for something specific?
[00:07.000 --> 00:13.000]  In the menu you can navigate to the overview of all public media of a facility.
[00:14.000 --> 00:17.000]  Do you want to browse? Then look in the video catalog.
[00:18.000 --> 00:22.000]  There you can filter by faculty, semester and category.
```

Note that the '--language' / '-l' parameter denotes the source language of the media file. Whisper's translation feature is always translating *into* English. Also note that if you want a subtitle in the original language *and* the English translation of it, you need to run Whisper/Subtitle2go two times. 

### What is an "initial prompt"?

You can use the initial prompt feature to provide additional context or a custom vocabulary (currently Whisper only). This will, for the most part, influence the beginning of decoding. For media files you could also use context information such as the title, author or speaker name's as initial prompt. For example:

```
$ python3 subtitle2go.py -e whisper --whisper-initial-prompt "Lecture2Go" --whisper-task "translate" subtitle2go-test.mp4
Using ASR beamsize: 5
Using Whisper as ASR engine.
self.filename='subtitle2go-test.mp4' self.fn_short_hash='17a23f553078047a' status='Starting Whisper decode.'
[00:01.000 --> 00:04.000]  Hello and welcome to Lecture2Go.
[00:05.000 --> 00:07.000]  Are you looking for something specific?
[00:07.000 --> 00:13.000]  In the menu you can navigate to the overview of all public media of a facility.
[00:14.000 --> 00:17.000]  Do you want to browse? Then look in the video catalog.
[00:18.000 --> 00:22.000]  There you can filter by faculty, semester and category.
```

### How do I use Whisper's automatic language detection?

You can set language to 'auto':

```
(subtitle2go_env) me@me-desktop:/scratch/projects/subtitle2go_v2$ python3 subtitle2go.py -e whisper -l auto --whisper-initial-prompt "Lecture2Go" --debug subtitle2go-test.mp4
Using ASR beamsize: 5
Using Whisper as ASR engine.
self.filename='subtitle2go-test.mp4' self.fn_short_hash='263c36edc2801fbb' status='Starting Whisper decode.'
Detecting language using up to the first 30 seconds. Use `--language` to specify the language
Detected language: German
[00:00.000 --> 00:08.880]  Hallo und willkommen bei Lecture2Go. Du suchst etwas Bestimmtes? Im Menü kannst du direkt zur
[00:08.880 --> 00:14.960]  Übersicht aller öffentlichen Medien einer Einrichtung navigieren. Willst du stöbern?
[00:14.960 --> 00:21.960]  Dann schau in den Videokatalog. Dort kannst du nach Fakultät, Semester und Kategorie filtern.
```

Note that only the first 30 seconds are used to determine the language and that the language detection doesn't have 100% accurarcy. 

### Help! I get repeating sentences or words with Whisper. Is this a bug?

Hallucinations and repetition loops can unfortunalty happen with the encoder/decoder architecture. Note that even the Open AI official Whisper API suffers from this problem.

You can try different settings, sometimes it helps to pass '--no-condition-on-previous-text'. Accurarcy might be slightly reduced, as no context information is passed between Whisper's 30 second processing blocks. Note that the initial prompt feature also stops working if you use this option.

### Error message ImportError: libkaldi-base.so: cannot open shared object file: No such file or directory


You have to load the path.sh into your shell. You need to run:

. path.sh

Every time you start a new shell session in order to use subtitle2go

### Error message "had nonzero return status 32512"
The path of kaldi is missing or incorrect
Use
```
export KALDI_ROOT=PATH_TO_KALDI
export PATH=$KALDI_ROOT/src/lmbin/:$KALDI_ROOT/../kaldi_lm/:$PWD/utils/:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lm/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnetbin:$KALDI_ROOT/src/nnet2bin/:$KALDI_ROOT/src/online2bin/:$KALDI_ROOT/src/ivectorbin/:$KALDI_ROOT/src/kwsbin:$KALDI_ROOT/src/nnet3bin:$KALDI_ROOT/src/chainbin:$KALDI_ROOT/tools/sph2pipe_v2.5/:$KALDI_ROOT/src/rnnlmbin:$PWD:$PATH
```
and replace PATH_TO_KALDI with the path to your kaldi binaries (eg. pykaldi/tools/kaldi)

## References

If you use Subtitle2Go in your academic work, please cite [this paper](http://www.essv.de/essv2021/pdfs/33_milde_v2.pdf):

    @InProceedings{milde-etal-21-subtitle2go,
      author={Benjamin Milde and Robert Geislinger and Irina Lindt and Timo Baumann},
      title={Open Source Automatic Lecture Subtitling},
      booktitle={Proceedings of ESSV 2021},
      year={2021},
      pages={128--134},
      address={Virtual Berlin, Germany}
    }
    
optionally also [this paper](https://aclanthology.org/2022.konvens-1.11/) for Subtitle2Go and the newest German ASR model:

     @InProceedings{geislinger-etal-2022-improved,
        title={Improved Open Source Automatic Subtitling for Lecture Videos},
        author={Robert Geislinger and Benjamin Milde and Chris Biemann},
        booktitle={Proceedings of the 18th Conference on Natural Language Processing (KONVENS 2022)},
        year={2022},
        address={Potsdam, Germany},
        pages={98--103}
     }

and for the previous German ASR model:

    @InProceedings{milde-koehn-18-german-asr,
      author={Benjamin Milde and Arne K{\"o}hn},
      title={Open Source Automatic Speech Recognition for German},
      booktitle={Proceedings of ITG 2018},
      year={2018},
      pages={251--255},
      address={Oldenburg, Germany}
    }

Thanks!
