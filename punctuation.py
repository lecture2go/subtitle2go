# Interpunctuation
from rpunct import RestorePuncts

# Adds interpunctuation to the Kaldi output
def interpunctuation(vtt, words, filenameS_hash, model_punctuation, uppercase, status):

    status.publish_status('Starting interpunctuation.')

    # BERT
    text = str(' '.join(words))
    if uppercase:
        text = text.lower()
    rpunct = RestorePuncts(model=model_punctuation)

    punct = rpunct.punctuate(text)
    punct = punct.replace('.', '. ').replace(',', ', ').replace('!', '! ').replace('?', '? ')
    punct = punct.replace('  ', ' ')
    punct_list = punct.split(' ')

    # punct_list = file_punct.read().split(' ')
    vtt_punc = []
    for a, b in zip(punct_list, vtt):  # Replaces the adapted words with the (capitalization, period, comma) with the new ones
        vtt_punc.append([a, b[1], b[2]])

    status.publish_status('Adding interpunctuation finished.')

    return vtt_punc