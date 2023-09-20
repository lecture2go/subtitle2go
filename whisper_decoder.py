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

from whisper.utils import format_timestamp

from typing import Iterator, TextIO

import sys
import whisper

# The write_vtt function was replaced in whisper, its a bit annoying
# this is the old version, copied from a previous version of whisper
# see https://github.com/openai/whisper/commit/da600abd2b296a5450770b872c3765d0a5a5c769
def write_vtt(transcript: Iterator[dict], file: TextIO):
    print("WEBVTT\n", file=file)
    for segment in transcript:
        print(
            f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
            f"{segment['text'].strip().replace('-->', '->')}\n",
            file=file,
            flush=True,
        )

def whisper_asr(filename, status, language, model='small', best_of=5, beam_size=5,
                condition_on_previous_text=True, fp16=True):
    if status:
        status.publish_status('Starting Whisper decode.')

    result = None

    # Filename without file extension
    filename_without_extension = filename.rpartition('.')[0]

    try:
        whisper_model = whisper.load_model(model)
        result = whisper_model.transcribe(filename, language=language, task='transcribe',
                                          temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                              best_of=best_of, beam_size=beam_size, suppress_tokens="-1",
                              condition_on_previous_text=condition_on_previous_text, fp16=fp16,
                              compression_ratio_threshold=2.4, logprob_threshold=-1., no_speech_threshold=0.6,
                              verbose=True, status=status)

        with open(filename_without_extension + '.vtt', 'w') as outfile:
            write_vtt(result["segments"], file=outfile)

    except Exception as e:
        if status:
            status.publish_status(f'Whisper decode failed.')
            status.publish_status(f'Error message is: {e}')
            status.send_error()
            sys.exit(-2)

    if status:
        status.publish_status(f'VTT finished. Model reported language: {result["language"]}')
    print('Done!')

    return result