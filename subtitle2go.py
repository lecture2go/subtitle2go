#!/usr/bin/env python
# -*- coding: utf-8 -*-

#    Copyright 2022 HITeC e.V., Benjamin Milde and Robert Geislinger
#    Copyright 2023 Lecture2Go, Dr. Benjamin Milde
#
#    Licensed under the Apache License, Version 2.0 (the 'License');
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an 'AS IS' BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from kaldi_decoder import asr, kaldi_time_to_seconds
import yaml
import argparse
import segment_text
import sys

from punctuation import interpunctuation
from utils import output_status, ensure_dir
from whisper_decoder import whisper_asr


# This creates a segmentation for the subtitles and make sure it can still be mapped to the Kaldi tokenisation
def vtt_segmentation(vtt, model_spacy, beam_size, ideal_token_len, len_reward_factor, comma_end_reward_factor,
                     sentence_end_reward_factor, status):
    sequences = []

    status.publish_status('Start text segmentation.')

    # Array starts at zero
    word_counter = -1
    
    # Makes a string for segmentation and change the <UNK> and <unk> Token to UNK
    word_string = ' '.join([e[0].replace('<UNK>', 'UNK').replace('<unk>', 'UNK') for e in vtt])
    
    # Call the segmentation beamsearch
    segments = segment_text.segment_beamsearch(word_string, model_spacy, beam_size=beam_size,
                                               ideal_token_len=ideal_token_len,
                                               len_reward_factor=len_reward_factor,
                                               sentence_end_reward_factor=sentence_end_reward_factor,
                                               comma_end_reward_factor=comma_end_reward_factor)

    temp_segments = [segments[0]]

    # Corrects punctuation marks and also lost tokens when they are slipped
    # to the beginning of the next line
    for current in segments[1:]:
        current_l = current.split(' ')
        if current_l[0].startswith((',', '.', '?', '!', "'", "n't")):
            temp_segments[-1] += current_l[0]
            current_l = current_l[1:]
        temp_segments.append(' '.join(current_l))
    segments = temp_segments

    # Cuts the segments in words, removes empty objects
    # and creates the sequences object
    for segment in segments:
        clean_segment = list(filter(None, segment.split(' ')))
        string_segment = ' '.join(clean_segment)
        segment_length = len(clean_segment)
        # Fixes problems with the first token. The first token is everytime 0
        if vtt[word_counter + 1][1] == 0:
            begin_segment = vtt[word_counter + 2][1]
        else:
            begin_segment = vtt[word_counter + 1][1]
        # this check is a workaround to not get index out a range error which may happen
        # (why? didn't want to get deep into the segmentation code)
        if (word_counter + segment_length) < len(vtt):
            end_segment = vtt[word_counter + segment_length][1] + vtt[word_counter + segment_length][2]
        else: 
            # use last segment as end_segment
            end_segment = vtt[-1][1] + vtt[-1][2] 
        sequences.append([string_segment, begin_segment, end_segment])
        word_counter = word_counter + segment_length
    
    status.publish_status('Text segmentation finished.')
    
    return sequences


# Creates the subtitle in the desired subtitleFormat and writes to filenameS (filename stripped) + subtitle suffix
def create_kaldi_subtitle(sequences, subtitle_format, filename_without_extension, status):
    status.publish_status('Start creating subtitle.')

    try:
        if subtitle_format == 'vtt':
            file = open(filename_without_extension + '.vtt', 'w')
            file.write('WEBVTT\n\n')
            separator = '.'
        elif subtitle_format == 'srt':
            file = open(filename_without_extension + '.srt', 'w')
            separator = ','
        else:
            status.publish_status(f"Output format: {subtitle_format} invalid!")
            status.send_error()
            sys.exit(-5)

        sequence_counter = 1
        for a in sequences:
            time_start = kaldi_time_to_seconds(a[1], separator)
            time_end = kaldi_time_to_seconds(a[2], separator)

            file.write(str(sequence_counter) + '\n')  # number of actual sequence

            timestring = time_start + ' --> ' + time_end + '\n'
            file.write(timestring)
            file.write(a[0] + '\n\n')
            sequence_counter += 1
        file.close()

    except Exception as e:
        status.publish_status('Subtitle creation failed.')
        status.publish_status(f'error message is: {e}')
        status.send_error()
        sys.exit(-1)

    status.publish_status('Finished subtitle creation.')


def pykaldi_subtitle(status, args, filename, filename_without_extension, filename_without_extension_hash):
    vtt, words = asr(filename_without_extension_hash, filename=filename, asr_beamsize=args.asr_beam_size,
                     asr_max_active=args.asr_max_active, acoustic_scale=args.acoustic_scale,
                     do_rnn_rescore=args.rnn_rescore, config_file=model_kaldi, status=status)
    vtt = interpunctuation(vtt, words, filename_without_extension_hash, model_punctuation, uppercase, status=status)
    sequences = vtt_segmentation(vtt, model_spacy, beam_size=args.segment_beam_size,
                                 ideal_token_len=args.ideal_token_len,
                                 len_reward_factor=args.len_reward_factor,
                                 sentence_end_reward_factor=args.sentence_end_reward_factor,
                                 comma_end_reward_factor=args.comma_end_reward_factor, status=status)
    create_kaldi_subtitle(sequences, subtitle_format, filename_without_extension, status=status)


if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--engine', help='The ASR engine to use. One of: kaldi, whisper or speechcatcher.',
                        required=False, default='kaldi', choices=['kaldi', 'whisper', 'speechcatcher'])

    # Flag (- and --) arguments
    parser.add_argument('-s', '--subtitle', help='The output subtitleformat (vtt or srt). Default=vtt',
                        required=False, default='vtt', choices=['vtt', 'srt'])

    parser.add_argument('-l', '--language', help='Sets the language of the models (de/en/...)',
                        required=False, default='de')

    parser.add_argument('-m', '--model-yaml', help='Kaldi model used for decoding (yaml config).',
                        type=str, default='models/kaldi_tuda_de_nnet3_chain2_de_683k.yaml')

    parser.add_argument('-i', '--id', help='Manually sets the file id', type=str,
                        required=False)

    parser.add_argument('-c', '--callback-url', help='Sets a callback URL to notify when process is'
                                                     ' finished or something went off', type=str,
                        required=False)

    parser.add_argument('--rnn-rescore', help='Do RNNLM rescoring of the decoder output (experimental,'
                                              ' needs more testing).',
                        action='store_true', default=False)

    parser.add_argument('--acoustic-scale', help='ASR decoder option: This is a scale on the acoustic'
                                                 ' log-probabilities, and is a universally used kludge'
                                                 ' in HMM-GMM and HMM-DNN systems to account for the'
                                                 ' correlation between frames.',
                        type=float, default=1.0)

    parser.add_argument('--asr-beam-size', help='ASR decoder option: controls the beam size in the beam search.'
                                                ' This is a speed / accuracy tradeoff.',
                        type=int, default=None)

    parser.add_argument('--asr-max-active', help='ASR decoder option: controls the maximum number of states that '
                                                 'can be active at one time.',
                        type=int, default=16000)

    parser.add_argument('--segment-beam-size', help='What beam size to use for the segmentation search',
                        type=int, default=10)
    parser.add_argument('--ideal-token-len', help='The ideal length of tokens per segment',
                        type=int, default=10)

    parser.add_argument('--len-reward-factor', help='How important it is to be close to ideal_token_len,'
                                                    ' higher factor = splits are closer to ideal_token_len',
                        type=float, default=2.3)
    parser.add_argument('--sentence-end-reward_factor', help='The weight of the sentence end score in the search.'
                                                             ' Higher values make it more likely to always split '
                                                             'at sentence end.',
                        type=float, default=0.9)
    parser.add_argument('--comma-end-reward-factor', help='The weight of the comma end score in the search. '
                                                          'Higher values make it more likely to'
                                                          ' always split at commas.',
                        type=float, default=0.5)

    parser.add_argument('--with-redis-updates', help='Update a redis instance about the current progress.',
                        action='store_true', default=False)

    parser.add_argument('--debug', help='Output debug timing information', default=False)

    # Positional argument, without (- and --)
    parser.add_argument('filename', help='The path of the mediafile', type=str)

    beamsize_default = {
        'kaldi': 13,
        'whisper': 5,
        'speechcatcher': 10
    }

    args = parser.parse_args()

    if args.asr_beam_size is None:
        args.asr_beam_size = beamsize_default.get(args.engine, 13)
        print("Using ASR beamsize:", args.asr_beam_size)

    filename = args.filename
    filename_without_extension = filename.rpartition('.')[0]
    subtitle_format = args.subtitle
    beamsize = args.asr_beam_size

    debug_word_timing = args.debug
    file_id = args.id
    callback_url = args.callback_url

    # The default is to use the hash of filename_without_extension as file id.
    # But it can also be set manually, if --id is used on the command line
    if file_id:
        filename_without_extension_hash = file_id
    else:
        filename_without_extension_hash = hex(abs(hash(filename_without_extension)))[2:]

    # Init status class
    status = output_status(redis=args.with_redis_updates, filename=filename,
                           fn_short_hash=filename_without_extension_hash, callback_url=callback_url)

    # Language selection
    language = args.language

    if args.engine == 'kaldi':
        ensure_dir('tmp/')
        with open('languages.yaml', 'r') as stream:
            language_yaml = yaml.safe_load(stream)
            if language_yaml.get(language, None):
                model_kaldi = language_yaml[language]['kaldi']
                model_punctuation = language_yaml[language]['punctuation']
                model_spacy = language_yaml[language]['spacy']
                uppercase = language_yaml[language]['uppercase']
            else:
                print(f'Language {language} is not set in languages.yaml. Exiting.')
                sys.exit()

        pykaldi_subtitle(status, args, filename, filename_without_extension, filename_without_extension_hash)
    elif args.engine == 'whisper':
        whisper_asr(filename, status, language, model='small', best_of=5, beam_size=beamsize,
                    condition_on_previous_text=True, fp16=True)
    else:
        print(args.engine, 'is not a valid engine.')

    status.publish_status('Job finished successfully.')
    status.send_success()
