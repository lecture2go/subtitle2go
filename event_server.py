#!/usr/bin/env python
# -*- coding: utf-8 -*-

#    Copyright 2022 HITeC e.V.
#    Copyright 2023 Lecture2Go
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

#

import argparse
import threading
import flask
import redis
import json

from werkzeug.serving import WSGIRequestHandler
from flask import jsonify, json, request, Response

import subprocess
import fcntl
import os
import signal
import psutil

__author__ = 'Dr. Benjamin Milde'

redis_server_channel = 'subtitle2go'

app = flask.Flask(__name__)
red = redis.StrictRedis(charset='utf-8', decode_responses=True)

long_poll_timeout = 0.5
long_poll_timeout_burst = 0.08

current_jobs = {}

def persistence_event_stream():
    global current_jobs
    print('Estabilishing persistence event_stream...')
    pubsub = red.pubsub()
    pubsub.subscribe(redis_server_channel)
    for message in pubsub.listen():
        msg = str(message['data'])
        print('New msg:', msg)
        if 'pid' in msg:
            msg_json = json.loads(msg)
            key = str(msg_json['pid']) + '_' + msg_json['file_id']
            current_jobs[key] = msg_json


def event_stream():
    print('New connection to event_stream!')
    pubsub = red.pubsub()
    pubsub.subscribe(redis_server_channel)
    #   yield b'hello'
    for message in pubsub.listen():
        if not message['type'] == 'subscribe':
            # print('New message:', message)
            # print(type(message['data']))
            yield b'data: %s\n\n' % message['data']


@app.route('/status')
def status():
    return jsonify(current_jobs)


@app.route('/status/<jobid>')
def status_with_id(jobid):
    if jobid in current_jobs:
        return jsonify(current_jobs[jobid])
    else:
        return jsonify({'error': 'could not find jobid in current jobs.'})


@app.route('/load')
def check_current_load():
    max_parallel_processes = 60

    subtitle2go_processes = 0
    for p in psutil.process_iter():
        if "subtitle2go.py" in "".join(p.cmdline()):
            subtitle2go_processes += 1

    response = dict()
    response['current_processes'] = subtitle2go_processes
    # true if free resources are available, false if not
    response['takes_job'] = subtitle2go_processes < max_parallel_processes

    return jsonify(response)

@app.route('/free_gpu', methods=['GET'])
def find_free_gpu_by_memory():
    try:
        # run nvidia-smi to get GPU memory usage
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'])
        memory_usages = [int(usage) for usage in result.strip().split('\n')]
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'])
        memory_totals = [int(total) for total in result.strip().split('\n')]
        free_memory = [total - used for total, used in zip(memory_totals, memory_usages)]
        # find the first GPU with >90% free memory
        for i, memory in enumerate(free_memory):
            if memory / memory_totals[i] > 0.9:
                return jsonify({'gpu_index': i})
        # no suitable GPU found, return -1
        return jsonify({'gpu_index': -1})
    except Exception as e:
        print("Error finding free GPU by memory:", str(e))
        return jsonify({'error': str(e)})

@app.route('/start', methods=['POST'])
def start():
    request_data = request.get_json()

    filename = request_data['filename']
    language = request_data['language']
    engine = 'kaldi'

    if 'engine' in request_data:
        engine = request_data['engine']

    callback_url = request_data['url'] + "/" + request_data['id']

    # calculate and return id
    filenameS = filename.rpartition('.')[0]  # Filename without file extension
    filenameS_hash = hex(abs(hash(filenameS)))[2:]

    # prepare logging
    log_file = filename + "_" + request_data['id'] + ".log"

    # if 'gpu_index' is provided in the request data, we need to set CUDA_VISIBLE_DEVICES.
    if 'gpu_index' in request_data:
        gpu_index = request_data['gpu_index']
        if gpu_index == -1:
            # force CPU usage by setting empty CUDA_VISIBLE_DEVICES
            # for gpu_index == -1, meaning no suitable free GPU available.
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_index)

    # run subtitle2go in background with the calculated id
    with open(log_file, "w+") as out:
        p = subprocess.Popen(
            ["python", "subtitle2go.py", "-e", engine, "-l", language, "--with-redis-updates", "-i", filenameS_hash,
             "-c", callback_url, filename], stdout=out, stderr=out)

    id = str(p.pid) + '_' + filenameS_hash
    return id


@app.route('/stop', methods=['POST'])
def stop():
    request_data = request.get_json()

    subtitle2goId = request_data['speech2TextId']

    pid = int(subtitle2goId.split("_")[0]);

    # kill the process if still running
    killed = False
    try:
        os.kill(pid, signal.SIGKILL)
    except Exception as e:
        pass
    else:
        killed = True
        del current_jobs[subtitle2goId]
    return str(killed)


@app.route('/clear')
def clear_finished():
    to_delete = []
    for key in current_jobs:
        if 'finished' in current_jobs[key]['status'] \
                or 'failed' in current_jobs[key]['status']:
            to_delete.append(key)
    for key in to_delete:
        del current_jobs[key]
    return 'ok'


# Event stream end point for the browser, connection is left open. Must be used with threaded Flask.
@app.route('/stream')
def stream():
    return flask.Response(event_stream(), mimetype='text/event-stream')


# Traditional long polling. This is the fall back, if a browser does not support server side events.
@app.route('/stream_poll')
def poll():
    pubsub = red.pubsub()
    pubsub.subscribe(redis_server_channel)
    message = pubsub.get_message(timeout=long_poll_timeout)
    while (message != None):
        yield message
        message = pubsub.get_message(timeout=long_poll_timeout_burst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Status update server for subtitle2go jobs')
    parser.add_argument('-l', '--listen-host', default='127.0.0.1', dest='host', help='Host address to listen on.')
    parser.add_argument('-p', '--port', default=7500, dest='port', help='Port to listen on.', type=int)
    parser.add_argument('--debug', dest='debug', help='Start with debugging enabled', action='store_true',
                        default=False)

    args = parser.parse_args()

    # print(' * Starting app with base path:',base_path)
    if args.debug:
        app.debug = True

    persistence_event_stream_thread = threading.Thread(target=persistence_event_stream)
    persistence_event_stream_thread.start()
    print('Running as testing server.')
    print('Host:', args.host)
    print('Port:', args.port)

    WSGIRequestHandler.protocol_version = 'HTTP/1.1'
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False, use_debugger=False)
