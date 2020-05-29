"""
Precompute ngram counts of captions, to accelerate cider computation during training time.
"""

import os
import json
import argparse
from six.moves import cPickle
import captioning.utils.misc as utils
from collections import defaultdict

import sys
sys.path.append("cider")
from pyciderevalcap.ciderD.ciderD_scorer import CiderScorer


def get_doc_freq(refs, params):
    tmp = CiderScorer(df_mode="corpus")
    for ref in refs:
        tmp.cook_append(None, ref)
    tmp.compute_doc_freq()
    return tmp.document_frequency, len(tmp.crefs)


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

    ngram_words, count_refs = get_doc_freq(refs_words, params)
    ngram_idxs, count_refs = get_doc_freq(refs_idxs, params)
    print('count_refs:', count_refs)
    return ngram_words, ngram_idxs, count_refs

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
    parser.add_argument('--input_json', default='data/dataset_coco.json', help='input json file to process into hdf5')
    parser.add_argument('--dict_json', default='data/cocotalk.json', help='output json file')
    parser.add_argument('--output_pkl', default='data/coco-all', help='output pickle file')
    parser.add_argument('--split', default='all', help='test, val, train, all')
    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict

    main(params)
