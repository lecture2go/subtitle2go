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

import yaml
import argparse
import segment_text
import sys

from utils import output_status, ensure_dir, format_timestamp_str


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
def create_subtitle(sequences, subtitle_format, filename_without_extension, convert_kaldi_time=False,
                    subtitle_offset=0.0, status=None):

    if status:
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
            if status:
                status.publish_status(f"Output format: {subtitle_format} invalid!")
                status.send_error()
            sys.exit(-5)

        sequence_counter = 1
        for a in sequences:
            time_start = format_timestamp_str(a[1], subtitle_offset, separator,
                                              convert_from_kaldi_time=convert_kaldi_time)
            time_end = format_timestamp_str(a[2], subtitle_offset, separator,
                                            convert_from_kaldi_time=convert_kaldi_time)


            file.write(str(sequence_counter) + '\n')  # number of actual sequence

            timestring = time_start + ' --> ' + time_end + '\n'
            file.write(timestring)
            file.write(a[0] + '\n\n')
            sequence_counter += 1
        file.close()

    except Exception as e:
        if status:
            status.publish_status('Subtitle creation failed.')
            status.publish_status(f'error message is: {e}')
            status.send_error()
        sys.exit(-1)

    status.publish_status('Finished subtitle creation.')


def pykaldi_subtitle(status, args, filename, filename_without_extension, filename_without_extension_hash,
                     subtitle_format):
    vtt, words = kaldi_asr(filename_without_extension_hash, filename=filename, asr_beamsize=args.asr_beam_size,
                           asr_max_active=args.asr_max_active, acoustic_scale=args.acoustic_scale,
                           do_rnn_rescore=args.rnn_rescore, config_file=model_kaldi, status=status)
    vtt = interpunctuation(vtt, words, filename_without_extension_hash, model_punctuation, uppercase, status=status)
    sequences = vtt_segmentation(vtt, model_spacy, beam_size=args.segment_beam_size,
                                 ideal_token_len=args.ideal_token_len,
                                 len_reward_factor=args.len_reward_factor,
                                 sentence_end_reward_factor=args.sentence_end_reward_factor,
                                 comma_end_reward_factor=args.comma_end_reward_factor, status=status)
    create_subtitle(sequences, subtitle_format, filename_without_extension, convert_kaldi_time=True,
                    subtitle_offset=args.subtitle_offset, status=status)


if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--engine', help='The ASR engine to use. One of: kaldi, whisper or speechcatcher.',
                        required=False, default='speechcatcher', choices=['speechcatcher', 'kaldi', 'whisper'])

    # Flag (- and --) arguments
    parser.add_argument('-s', '--subtitle', help='The output subtitleformat (vtt or srt). Default=vtt',
                        required=False, default='vtt', choices=['vtt', 'srt'])

    parser.add_argument('-l', '--language', help='Sets the language of the models (de/en/...).'
                                                 'With the engine option set to "whisper" you'
                                                 ' can also use "auto" for automatic language'
                                                 ' detection.',
                        required=False, default='de')

    parser.add_argument('-m', '--model-yaml', help='Model used for decoding (yaml config for'
                                                   ' kaldi or model name for other engines).',
                        type=str, default=None)

    parser.add_argument('-i', '--id', help='Manually sets the file id', type=str,
                        required=False)

    parser.add_argument('-c', '--callback-url', help='Sets a callback URL to notify when process is'
                                                     ' finished or something went off', type=str,
                        required=False)

    parser.add_argument('-p', '--num-procs', help='Number of parallel processors (Speechcatcher and Whisper only)',
                        type=int, default=-1)

    parser.add_argument('-o', '--subtitle-offset', help='Subtitle offset (in seconds)',
                        type=float, default=None)

    parser.add_argument('--rnn-rescore', help='Do RNNLM rescoring of the decoder output (only for PyKaldi'
                                              'models).',
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

    # Whisper specific options:
    parser.add_argument('--whisper-task', help='The whisper task: one of either "transcribe" or "translate".',
                        required=False, default='transcribe', choices=['transcribe', 'translate'])

    parser.add_argument('--no-condition-on-previous-text', help='Disabling condition-on-previous-text will'
                                                                ' reduce Whispers accuracy, but can sometimes help to '
                                                                'avoid hallucinations.',
                        action='store_true', default=False)

    parser.add_argument('--whisper-initial-prompt', help='Initial prompt for the first segment. Can be used to pass in'
                                                         'useful additional information like an author name, a title,'
                                                         ' a custom vocabulary etc.',
                        type=str, default=None)

    parser.add_argument('--whisper-no-speech-threshold', help='Threshold parameter to decide if a segment is speech'
                                                              'or not speech. Default is 0.6.',
                        type=float, default=0.6)

    parser.add_argument('--with-redis-updates', help='Update a redis instance about the current progress.',
                        action='store_true', default=False)

    parser.add_argument('--debug', help='Output debug timing information', action='store_true', default=False)

    # Positional argument, without (- and --)
    parser.add_argument('filename', help='The path of the mediafile', type=str)

    beamsize_default = {
        'kaldi': 13,
        'whisper': 5,
        'speechcatcher': 10
    }

    engine_model_default = {
        'kaldi':  'models/kaldi_tuda_de_nnet3_chain2_de_900k.yaml',
        'speechcatcher': 'de_streaming_transformer_xl',
        'whisper': 'large-v2'
   }

    subtitle_offset_default = {
        'kaldi': 0.0,
        'speechcatcher': -0.1,
        'whisper': 0.0
    }

    args = parser.parse_args()

    if args.model_yaml is None:
        args.model_yaml = engine_model_default[args.engine]

    if args.asr_beam_size is None:
        args.asr_beam_size = beamsize_default.get(args.engine, 13)
        print("Using ASR beamsize:", args.asr_beam_size)

    if args.subtitle_offset is None:
        args.subtitle_offset = subtitle_offset_default.get(args.engine, 0.0)

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
        # dynamic import
        from kaldi_decoder import kaldi_asr
        from punctuation import interpunctuation

        print("Using Kaldi as ASR engine.")
        ensure_dir('tmp/')
        with open('kaldi_languages.yaml', 'r') as stream:
            language_yaml = yaml.safe_load(stream)
            if language_yaml.get(language, None):
                model_kaldi = language_yaml[language]['kaldi']
                model_punctuation = language_yaml[language]['punctuation']
                model_spacy = language_yaml[language]['spacy']
                uppercase = language_yaml[language]['uppercase']
            else:
                print(f'Language {language} is not set in kaldi_languages.yaml. Exiting.')
                sys.exit()

        pykaldi_subtitle(status, args, filename, filename_without_extension, filename_without_extension_hash,
                         subtitle_format)
    elif args.engine == 'whisper':
        # dynamic import
        import torch
        # set num_threads according to how many parallel processors we should use
        if args.num_procs > 0:
            torch.set_num_threads(args.num_procs)
        from whisper_decoder import whisper_asr

        print("Using Whisper as ASR engine.")

        # Setting language to None means that Whisper will use the first
        # 30 seconds to automatically determine the language
        if language == 'auto':
            language = None
        whisper_asr(filename, status=status, task=args.whisper_task, language=language, output_format=args.subtitle,
                    model=args.model_yaml, best_of=5,
                    beam_size=beamsize, initial_prompt=args.whisper_initial_prompt,
                    condition_on_previous_text=not args.no_condition_on_previous_text,
                    fp16=True, no_speech_threshold=args.whisper_no_speech_threshold, verbose=args.debug)
    elif args.engine == 'speechcatcher':
        # dynamic import
        import torch
        # Note that we need to set this to 1, otherwise the decoding will hang with num_procs > 1.
        # It seems that torch threads are interfering with Speechcatcher's
        # parallelization (ProcessPoolExecutor with concurrent.futures).
        torch.set_num_threads(1)
        from speechcatcher_decoder import speechcatcher_asr, speechcatcher_vtt_segmentation

        print("Using Speechcatcher as ASR engine.")

        with open('kaldi_languages.yaml', 'r') as stream:
            language_yaml = yaml.safe_load(stream)
            if language_yaml.get(language, None):
                model_spacy = language_yaml[language]['spacy']
            else:
                print(f'Language {language} is not set in kaldi_languages.yaml. Exiting.')
                sys.exit()

        # The Speechcatcher srt/vtt output is generated in 3 steps;
        # (1) End-to end ASR (2) Segmentation and alignment of token time stamps to the segmented text
        # (3) Generate a VTT or SRT from the segments
        complete_text, paragraphs = speechcatcher_asr(filename, status, language=language,
                                                      model_short_tag=args.model_yaml, num_processes=args.num_procs)

        sequences = speechcatcher_vtt_segmentation(paragraphs, model_spacy, beam_size=args.segment_beam_size,
                                                   ideal_token_len=args.ideal_token_len,
                                                   len_reward_factor=args.len_reward_factor,
                                                   sentence_end_reward_factor=args.sentence_end_reward_factor,
                                                   comma_end_reward_factor=args.comma_end_reward_factor,
                                                   status=status)

        create_subtitle(sequences, subtitle_format, filename_without_extension, convert_kaldi_time=False,
                        subtitle_offset=args.subtitle_offset, status=status)
    else:
        print(args.engine, 'is not a valid engine.')

    if status:
        status.publish_status('Job finished successfully.')
        status.send_success()
