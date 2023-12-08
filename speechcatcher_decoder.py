#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

import hashlib
import wave
import os
import numpy as np
import multiprocessing
import segment_text
import sys
import traceback
from speechcatcher import speechcatcher


def speechcatcher_vtt_segmentation(paragraphs, model_spacy, beam_size, ideal_token_len, len_reward_factor,
                                   comma_end_reward_factor, sentence_end_reward_factor, status=None):

    num_warnings = 0

    if status:
        status.publish_status("Running subtitle segmentation...")
    sequences = []
    for paragraph in paragraphs:
        try:
            segments = segment_text.segment_beamsearch(paragraph["text"], model_spacy, beam_size=beam_size,
                                               ideal_token_len=ideal_token_len,
                                               len_reward_factor=len_reward_factor,
                                               sentence_end_reward_factor=sentence_end_reward_factor,
                                               comma_end_reward_factor=comma_end_reward_factor)
        except Exception as e:
            traceback.print_exc()
            num_warnings += 1
            warn_msg = f'Warning, paragraph segmentation failed. Exception: {type(e).__name__}'
            print(warn_msg)
            if status:
                status.publish_status(warn_msg)
                status.send_warning()
            continue

        tokens = paragraph["tokens"]
        token_timestamps = paragraph["token_timestamps"]

        start_token_idx = 0
        end_token_idx = 0

        try:
            for segment in segments:
                # iterate through tokens to find the start and end indices
                # note that the start position should be the end of the last segment
                start_token_idx = end_token_idx
                remaining_text = segment

                # match the segment to the tokens, so that we can get start and end positions
                # of the segment from the token timestamps
                while remaining_text:
                    # espnet uses '▁' (Unicode U+2581 Lower One Eighth Block Unicode Character) to denote a space
                    # in a token.
                    token = tokens[end_token_idx].replace('▁',' ')
                    if remaining_text.startswith(token):
                        remaining_text = remaining_text[len(token):]
                        end_token_idx += 1
                    elif remaining_text.startswith(token.replace(' ','')):
                        remaining_text = remaining_text[len(token.replace(' ','')):]
                        end_token_idx += 1
                    else:
                        # kinda bad handling if the segment doesn't perfectly match token boundaries,
                        # but we can try to advance the tokens, but something is probably broken if this happens:
                        end_token_idx += 1
                        print("Warning, segment overflow")
                        num_warnings += 1

                # get the timestamps for the start and end tokens
                start_timestamp = token_timestamps[start_token_idx]
                end_timestamp = token_timestamps[end_token_idx - 1]
                segment_info = (segment, start_timestamp, end_timestamp)

                sequences.append(segment_info)
        except (IndexError, AttributeError, TypeError, RuntimeError) as e:
            traceback.print_exc()
            num_warnings += 1
            warn_msg = f'Warning, segment/token alignment failed. Exception: {type(e).__name__}'
            print(warn_msg)
            if status:
                status.publish_status(warn_msg)
                status.send_warning()
            continue
    if status:

        status.publish_status("Finished subtitle segmentation." +
                              f" Note: there were {num_warnings} warnings, subtitles may be incomplete."
                              if num_warnings > 0 else "")
    return sequences


def speechcatcher_asr(media_path, status, language=None,
                      model_short_tag='de_streaming_transformer_xl',
                      chunk_length=8192, num_processes=-1, tmp_dir='./tmp/'):

    if language is not None and language != '' and language != 'auto' and language != 'ignore':
        if language not in model_short_tag:
            error_msg = f"Error, speechcatcher model {model_short_tag} seems to be" \
                        f" incompatible with language {language}."
            print(error_msg)
            if status:
                status.publish_status(error_msg)
            sys.exit(-5)

    if num_processes == -1:
        num_processes = multiprocessing.cpu_count() // 2

    if status:
        status.publish_status(f'Loading model {model_short_tag}...')

    try:
        speech2text = speechcatcher.load_model(speechcatcher.tags[model_short_tag])
    except Exception as e:
        traceback.print_exc()
        status.publish_status(f'Error, could not load Speechcatcher model. Error message is: {e}')
        status.send_error()
        sys.exit(-8)

    if status:
        status.publish_status('Converting input file to 16kHz mono audio...')

    speechcatcher.ensure_dir(tmp_dir)
    wavfile_path = tmp_dir + hashlib.sha1(media_path.encode("utf-8")).hexdigest() + '.wav'

    try:
        speechcatcher.convert_inputfile(media_path, wavfile_path)
    except Exception as e:
        traceback.print_exc()
        status.publish_status(f'Error, could not read and/or convert input media file. Error message is: {e}')
        status.send_error()
        sys.exit(-9)

    with wave.open(wavfile_path, 'rb') as wavfile_in:
        ch = wavfile_in.getnchannels()
        bits = wavfile_in.getsampwidth()*8
        rate = wavfile_in.getframerate()
        nframes = wavfile_in.getnframes()
        buf = wavfile_in.readframes(-1)
        raw_speech_data = np.frombuffer(buf, dtype='int16')

    assert(ch == 1)
    assert(bits == 16)
    assert(rate == 16000)

    if status:
        status.publish_status('Starting decoding with Speechcatcher model'
                              f' {model_short_tag} with {num_processes} processes.')

    try:
        # speech is a numpy array of dtype='np.int16' (16bit audio with 16kHz sampling rate)
        complete_text, paragraphs = speechcatcher.recognize(speech2text, raw_speech_data, rate, chunk_length=chunk_length,
                                                            num_processes=num_processes, progress=False,
                                                            quiet=True, status=status)
    except Exception as e:
        traceback.print_exc()
        status.publish_status(f'Error, could not decode speech with Speechcatcher. Error message is: {e}')
        status.send_error()
        sys.exit(-10)

    os.remove(wavfile_path)

    if status:
        status.publish_status('Finished decoding.')

    return complete_text, paragraphs