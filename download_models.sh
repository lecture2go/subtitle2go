#!/bin/sh

cd models/

# de kaldi models:
wget http://ltdata1.informatik.uni-hamburg.de/kaldi_tuda_de/de_900k_nnet3chain_tdnn1f_2048_sp_bi.tar.bz2
wget http://ltdata1.informatik.uni-hamburg.de/kaldi_tuda_de/de_900k_G.carpa.bz2
wget http://ltdata1.informatik.uni-hamburg.de/kaldi_tuda_de/de_900k_rnnlm_lstm_4x.tar.bz2

tar xvfj de_900k_nnet3chain_tdnn1f_2048_sp_bi.tar.bz2
bunzip2 de_900k_G.carpa.bz2
tar xvfj de_900k_rnnlm_lstm_4x.tar.bz2

# en kaldi models:
wget https://ltdata1.informatik.uni-hamburg.de/subtitle2go/en_200k_nnet3chain_tdnn1f_2048_sp_bi.tar.bz2

tar xvfj en_200k_nnet3chain_tdnn1f_2048_sp_bi.tar.bz2

# de punctuation models:
# wget https://ltdata1.informatik.uni-hamburg.de/subtitle2go/Model_subs_norm1_filt_5M_tageschau_euparl_h256_lr0.02.pcl
wget http://ltdata1.informatik.uni-hamburg.de/subtitle2go/interpunct_de_rpunct.tar.gz
tar xfvz interpunct_de_rpunct.tar.gz

# en punctuation models:
mkdir interpunct_en_rpunct
wget http://ltdata1.informatik.uni-hamburg.de/subtitle2go/interpunct_en_rpunct.tar.gz
tar xfvz interpunct_en_rpunct.tar.gz --directory interpunct_en_rpunct/

cd ..

python3 -m spacy download de_core_news_lg
python3 -m spacy download en_core_web_lg
