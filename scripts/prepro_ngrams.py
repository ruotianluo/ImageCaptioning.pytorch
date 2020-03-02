"""
Precompute ngram counts of captions, to accelerate cider computation during training time.
"""

import os
import json
import argparse
from six.moves import cPickle
import misc.utils as utils
from collections import defaultdict

def precook(s, n=4, out=False):
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    words = s.split()
    counts = defaultdict(int)
    for k in range(1,n+1):
        for i in range(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    return counts

def cook_refs(refs, n=4): ## lhuang: oracle will call with "average"
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    '''
    return [precook(ref, n) for ref in refs]

def create_crefs(refs):
    crefs = []
    for ref in refs:
        # ref is a list of 5 captions
        crefs.append(cook_refs(ref))
    return crefs

def compute_doc_freq(crefs):
    '''
    Compute term frequency for reference data.
    This will be used to compute idf (inverse document frequency later)
    The term frequency is stored in the object
    :return: None
    '''
    document_frequency = defaultdict(float)
    for refs in crefs:
        # refs, k ref captions of one image
        for ngram in set([ngram for ref in refs for (ngram,count) in ref.items()]):
            document_frequency[ngram] += 1
            # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)
    return document_frequency

def build_dict(imgs, wtoi, params):
    wtoi['<eos>'] = 0

    count_imgs = 0

    refs_words = []
    refs_idxs = []
    for img in imgs:
        if (params['split'] == img['split']) or \
            (params['split'] == 'train' and img['split'] == 'restval') or \
            (params['split'] == 'all'):
            #(params['split'] == 'val' and img['split'] == 'restval') or \
            ref_words = []
            ref_idxs = []
            for sent in img['sentences']:
                if hasattr(params, 'bpe'):
                    sent['tokens'] = params.bpe.segment(' '.join(sent['tokens'])).strip().split(' ')
                tmp_tokens = sent['tokens'] + ['<eos>']
                tmp_tokens = [_ if _ in wtoi else 'UNK' for _ in tmp_tokens]
                ref_words.append(' '.join(tmp_tokens))
                ref_idxs.append(' '.join([str(wtoi[_]) for _ in tmp_tokens]))
            refs_words.append(ref_words)
            refs_idxs.append(ref_idxs)
            count_imgs += 1
    print('total imgs:', count_imgs)

    ngram_words = compute_doc_freq(create_crefs(refs_words))
    ngram_idxs = compute_doc_freq(create_crefs(refs_idxs))
    return ngram_words, ngram_idxs, count_imgs

def main(params):

    imgs = json.load(open(params['input_json'], 'r'))
    dict_json = json.load(open(params['dict_json'], 'r'))
    itow = dict_json['ix_to_word']
    wtoi = {w:i for i,w in itow.items()}

    # Load bpe
    if 'bpe' in dict_json:
        import tempfile
        import codecs
        codes_f = tempfile.NamedTemporaryFile(delete=False)
        codes_f.close()
        with open(codes_f.name, 'w') as f:
            f.write(dict_json['bpe'])
        with codecs.open(codes_f.name, encoding='UTF-8') as codes:
            bpe = apply_bpe.BPE(codes)
        params.bpe = bpe

    imgs = imgs['images']

    ngram_words, ngram_idxs, ref_len = build_dict(imgs, wtoi, params)

    utils.pickle_dump({'document_frequency': ngram_words, 'ref_len': ref_len}, open(params['output_pkl']+'-words.p','wb'))
    utils.pickle_dump({'document_frequency': ngram_idxs, 'ref_len': ref_len}, open(params['output_pkl']+'-idxs.p','wb'))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', default='/home-nfs/rluo/rluo/nips/code/prepro/dataset_coco.json', help='input json file to process into hdf5')
    parser.add_argument('--dict_json', default='data/cocotalk.json', help='output json file')
    parser.add_argument('--output_pkl', default='data/coco-all', help='output pickle file')
    parser.add_argument('--split', default='all', help='test, val, train, all')
    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict

    main(params)
