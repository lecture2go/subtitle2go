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

import os
import time
import json
import requests
import ffmpeg

kaldi_feature_factor = 3. #used to be 3.00151874884282680911

# status object for sending status messages through redis and callbacks
class output_status():
    def __init__(self, filename, fn_short_hash, redis=False, callback_url=None):
        self.start_time = time.time()
        if redis:
            try:
                import redis
                self.red = redis.StrictRedis(charset='utf-8', decode_responses=True)
            except ImportError:
                print('Redis is not available. Disabling redis option.')
                redis = False

            self.redis_server_channel = 'subtitle2go'
        self.redis = redis

        self.filename = filename
        self.fn_short_hash = fn_short_hash

        self.callback_url = callback_url

    def publish_status(self, status):
        print(f'{self.filename=} {self.fn_short_hash=} {status=}')
        if self.redis:
            self.red.publish(self.redis_server_channel, json.dumps({'pid': os.getpid(), 'time': time.time(),
                                                                    'start_time': self.start_time,
                                                    'file_id': self.fn_short_hash, 'filename': self.filename,
                                                    'status': status}))

    def send_error(self):
        if (self.callback_url):
            json_data = {'message': 'false'}
            r = requests.put(self.callback_url, data=json_data)

    def send_warning(self):
        if (self.callback_url):
            json_data = {'message': 'false'}
            r = requests.put(self.callback_url, data=json_data)

    def send_success(self):
        if (self.callback_url):
            json_data = {'message': 'true'}
            r = requests.put(self.callback_url, data=json_data)

# Make sure a fpath directory exists
def ensure_dir(fpath):
    directory = os.path.dirname(fpath)
    if not os.path.exists(directory):
        os.makedirs(directory)


def preprocess_audio(filename, wav_filename):
    # Use ffmpeg to convert the input media file (any format!) to 16 kHz wav mono
    (
        ffmpeg
            .input(filename)
            .output(wav_filename, acodec='pcm_s16le', ac=1, ar='16k')
            .overwrite_output()
            .run(quiet=True)
    )


def format_timestamp_str(time, seperator, convert_from_kaldi_time=True):
    if convert_from_kaldi_time:
        time = time * kaldi_feature_factor / 100
    time_start = (f'{int(time / 3600):02}:'
                            f'{int(time / 60 % 60):02}:'
                            f'{int(time % 60):02}'
                            f'{seperator}'
                            f'{int(time * 1000 % 1000):03}')
    return time_start