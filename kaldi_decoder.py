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
import sys
import ffmpeg
import yaml

kaldi_feature_factor = 3.00151874884282680911

# Load Kaldi
from kaldi.asr import NnetLatticeFasterRecognizer, LatticeLmRescorer, LatticeRnnlmPrunedRescorer
from kaldi.rnnlm import RnnlmComputeStateComputationOptions
from kaldi.decoder import LatticeFasterDecoderOptions
from kaldi.lat.functions import ComposeLatticePrunedOptions
from kaldi.lat.align import read_lexicon_for_word_align
from kaldi.fstext import SymbolTable, shortestpath, indices_to_symbols
from kaldi.fstext.utils import get_linear_symbol_sequence
from kaldi.nnet3 import NnetSimpleComputationOptions
from kaldi.util.table import SequentialMatrixReader
from kaldi.lat import functions
from kaldi.transform import cmvn

#from subtitle2go import status, kaldi_time_to_seconds, debug_word_timing, preprocess_audio, send_error, args
from simple_endpointing import process_wav

from utils import *

def kaldi_time_to_seconds(value, seperator):
    time = value * kaldi_feature_factor / 100
    time_start =    (f'{int(time / 3600):02}:'
                            f'{int(time / 60 % 60):02}:'
                            f'{int(time % 60):02}'
                            f'{seperator}'
                            f'{int(time * 1000 % 1000):03}')
    return time_start

def recognizer(decoder_yaml_opts, models_dir):
    decoder_opts = LatticeFasterDecoderOptions()
    decoder_opts.beam = decoder_yaml_opts['beam']
    decoder_opts.max_active = decoder_yaml_opts['max-active']
    decoder_opts.lattice_beam = decoder_yaml_opts['lattice-beam']

    decodable_opts = NnetSimpleComputationOptions()
    decodable_opts.acoustic_scale = decoder_yaml_opts['acoustic-scale']
    decodable_opts.frame_subsampling_factor = 3 # decoder_yaml_opts['frame-subsampling-factor'] # 3
    decodable_opts.frames_per_chunk = 150
    fr = NnetLatticeFasterRecognizer.from_files(
        models_dir + decoder_yaml_opts['model'],
        models_dir + decoder_yaml_opts['fst'],
        models_dir + decoder_yaml_opts['word-syms'],
        decoder_opts=decoder_opts, decodable_opts=decodable_opts)

    return fr

# This method contains all Kaldi related calls and methods
def Kaldi(config_file, scp_filename, spk2utt_filename, segments_filename, do_rnn_rescore,
          segments_timing, lm_scale, acoustic_scale, status, debug_word_timing=False):

    models_dir = 'models/'

    # Read yaml File
    with open(config_file, 'r') as stream:
        model_yaml = yaml.safe_load(stream)
    decoder_yaml_opts = model_yaml['decoder']

    # Construct recognizer
    fr = recognizer(decoder_yaml_opts, models_dir)

    # Check if cmvn is set
    cmvn_transformer = None
    if decoder_yaml_opts.get('global-cmvn-stats'):
        cmvn_transformer = cmvn.Cmvn(40)
        cmvn_transformer.read_stats(f'{models_dir}{decoder_yaml_opts["global-cmvn-stats"]}')

    # Construct symbol table
    symbols = SymbolTable.read_text(models_dir + decoder_yaml_opts['word-syms'])

    # Define feature pipelines as Kaldi rspecifiers
    feats_rspec = (f'ark:extract-segments scp,p:{scp_filename} {segments_filename} '
                   f'ark:- | compute-mfcc-feats --config={models_dir}{decoder_yaml_opts["mfcc-config"]} ark:- ark:- |')

    ivectors_rspec = (
            (f'ark:extract-segments scp,p:{scp_filename} {segments_filename} '
             f'ark:- | compute-mfcc-feats --config={models_dir}{decoder_yaml_opts["mfcc-config"]} '
             f'ark:- ark:- | '
             f'ivector-extract-online2 --config={models_dir}{decoder_yaml_opts["ivector-extraction-config"]} '
             f'ark:{spk2utt_filename} ark:- ark:- |'))


    rnn_rescore_available = 'rnnlm' in decoder_yaml_opts

    if do_rnn_rescore and not rnn_rescore_available:
        if status:
            status.publish_status("Warning, disabling RNNLM rescoring since 'rnnlm'"
                              " is not in the decoder options of the .yaml config.")

    if do_rnn_rescore and rnn_rescore_available:
        status.publish_status('Loading language model rescorer.')
        rnn_lm_folder = models_dir + decoder_yaml_opts['rnnlm']
        arpa_G = models_dir + decoder_yaml_opts['arpa']
        old_lm = models_dir + decoder_yaml_opts['fst']

        print(f'Loading RNNLM rescorer from:{rnn_lm_folder} with ARPA from:{arpa_G} FST:{old_lm}')
        # Construct RNNLM rescorer
        symbols = SymbolTable.read_text(rnn_lm_folder+'/config/words.txt')
        rnnlm_opts = RnnlmComputeStateComputationOptions()
        rnnlm_opts.bos_index = symbols.find_index('<s>')
        rnnlm_opts.eos_index = symbols.find_index('</s>')
        rnnlm_opts.brk_index = symbols.find_index('<brk>')
        compose_opts = ComposeLatticePrunedOptions()
        compose_opts.lattice_compose_beam = 6
        print(f'rnnlm-get-word-embedding {rnn_lm_folder}/word_feats.txt {rnn_lm_folder}/feat_embedding.final.mat -|')
        print(f'{rnn_lm_folder}/final.raw')
        rescorer = LatticeRnnlmPrunedRescorer.from_files(
            arpa_G,
            f'rnnlm-get-word-embedding {rnn_lm_folder}/word_feats.txt {rnn_lm_folder}/feat_embedding.final.mat -|',
            f'{rnn_lm_folder}/final.raw', lm_scale=lm_scale, acoustic_scale=acoustic_scale, max_ngram_order=4,
            use_const_arpa=True, opts=rnnlm_opts, compose_opts=compose_opts)

    did_decode = False
    decoding_results = []

    segmentcounter = 1
    with SequentialMatrixReader(feats_rspec) as f, \
            SequentialMatrixReader(ivectors_rspec) as i:
            for (fkey, feats), (ikey, ivectors) in zip(f, i):
                # Calculate progress percentage
                progress_percentage = (segmentcounter / len(segments_timing)) * 100
                progress_message = f'Decoding progress: {progress_percentage:.2f}%'
                status.publish_status(progress_message)

                if cmvn_transformer:
                    cmvn_transformer.apply(feats)
                did_decode = True
                assert (fkey == ikey)
                out = fr.decode((feats, ivectors))
                if do_rnn_rescore:
                    lat = rescorer.rescore(out['lattice'])
                else:
                    lat = out['lattice']
                best_path = functions.compact_lattice_shortest_path(lat)
                words, _, _ = get_linear_symbol_sequence(shortestpath(best_path))
                timing = functions.compact_lattice_to_word_alignment(best_path)
                decoding_results.append((words, timing))
                segmentcounter+=1

    # Concatenating the results of the segments and adding an offset to the segments
    words = []
    timing = [[],[],[]]
    for result in decoding_results:
        words.extend(result[0])

    for result, offset in zip(decoding_results, segments_timing):
        if result[1][1]:
            timing[0].extend(result[1][0])
            # start = map(lambda x: int(x + (offset[0] / kaldi_feature_factor)), result[1][1])
            start = [x + (offset[0] / kaldi_feature_factor) for x in result[1][1]]
            kaldi_time_to_seconds(offset[0] / kaldi_feature_factor, seperator=".")
            kaldi_time_to_seconds(start[0], seperator=".")

            timing[1].extend(start)
            timing[2].extend(result[1][2])
    starting = 0
    temp_timing = [[], [], []]

    # Maps words to the numbers
    words = indices_to_symbols(symbols, timing[0])

    # Creates the datastructure (Word, begin(Frames), end(Frames))
    assert(len(words) == len(timing[1]))
    assert(len(timing[1]) == len(timing[2]))
    vtt = list(map(list, zip(words, timing[1], timing[2])))

    if debug_word_timing:
        with open('debug_output.txt', 'w') as f:
            for element in vtt:
                f.write(f'{element[1]} {kaldi_time_to_seconds(element[1], ".")}'
                        f' {kaldi_time_to_seconds(element[1] + element[2], ".")} {element[2]} {element[0]}\n')

    return vtt, did_decode, words

# This is the asr function that converts the videofile, split the video into segments and decodes
def asr(filenameS_hash, filename, asr_beamsize=13, asr_max_active=8000, acoustic_scale=1.0, lm_scale=0.5,
         do_rnn_rescore=False, config_file='models/kaldi_tuda_de_nnet3_chain2_de_722k.yaml', status=None):

    print(f"{filenameS_hash=}")

    scp_filename = f'tmp/{filenameS_hash}.scp'
    segments_filename = f'tmp/{filenameS_hash}_segments'
    wav_filename = f'tmp/{filenameS_hash}.wav'
    spk2utt_filename = f'tmp/{filenameS_hash}_spk2utt'

    # Audio extraction
    if status:
        status.publish_status('Extract audio.')

    try:
        preprocess_audio(filename, wav_filename)
    except ffmpeg.Error as e:
        if status:
            status.publish_status('Audio extraction failed.')
            status.publish_status(f'Error message is: {e.stderr}')
            status.send_error()
        print(f'Audio extraction failed: {e.stderr}')
        sys.exit(-1)

    if status:
        status.publish_status('Audio extracted.')
        status.publish_status('Audio segmentation.')

    # Segmentation

    try:
        segments_filenames, segments_timing = process_wav(wav_filename)
    except Exception as e:
        if status:
            status.publish_status('Audio segmentation failed.')
            status.publish_status(f'Error message is: {e}')
            status.send_error()
        print(f'Audio segmentation failed. {e.stderr}')
        sys.exit(-1)

    # Write scp and spk2utt file
    with open(scp_filename, 'w') as wavscp, open(spk2utt_filename, 'w') as spk2utt:
        # segmentFilename = wav_filename.rpartition('.')[0]
        wavscp.write(f'{filenameS_hash} {wav_filename}\n')

        for i in range(len(segments_timing)):
            count_str = "%.4d" % i
            spk2utt.write(f'{filenameS_hash} {filenameS_hash}_{count_str}\n')

    # Decode wav files
    if status:
        status.publish_status('Start ASR.')
    vtt, did_decode, words = Kaldi(config_file, scp_filename, spk2utt_filename, segments_filename,
                                   do_rnn_rescore, segments_timing, lm_scale, acoustic_scale, status)

    # communicate back job status
    if did_decode:
        if status:
            status.publish_status('ASR finished.')
    else:
        if status:
            status.publish_status('ASR error.')
            status.send_error()
        print('ASR error.')
        sys.exit(-1)

    # Cleanup tmp files
    try:
        os.remove(scp_filename)
        os.remove(wav_filename)
        os.remove(spk2utt_filename)
        os.remove(segments_filename)
        if status:
            status.publish_status(f'Removed temporary files: {scp_filename=}, {wav_filename=},'
                              f' {spk2utt_filename=}, {segments_filename=}')
    except Exception as e:
        if status:
            status.publish_status(f'Removing files failed.')
            status.publish_status(f'Error message is: {e}')
            status.send_error()
        print(f'Removing files failed: {e}')

    if status:
        status.publish_status('VTT finished.')

    return vtt, words