import os
import time
import json
import requests
import ffmpeg

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