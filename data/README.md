# Prepare data

## COCO

### Download COCO captions and preprocess them

Download preprocessed coco captions from [link](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) from Karpathy's homepage. Extract `dataset_coco.json` from the zip file and copy it in to `data/`. This file provides preprocessed captions and also standard train-val-test splits.

Then do:

```bash
$ python scripts/prepro_labels.py --input_json data/dataset_coco.json --output_json data/cocotalk.json --output_h5 data/cocotalk
```

`prepro_labels.py` will map all words that occur <= 5 times to a special `UNK` token, and create a vocabulary for all the remaining words. The image information and vocabulary are dumped into `data/cocotalk.json` and discretized caption data are dumped into `data/cocotalk_label.h5`.

### Download COCO dataset and pre-extract the image features (Skip if you are using bottom-up feature)

Download the coco images from [link](http://mscoco.org/dataset/#download). We need 2014 training images and 2014 val. images. You should put the `train2014/` and `val2014/` in the same directory, denoted as `$IMAGE_ROOT`.

Then:

```
$ python scripts/prepro_feats.py --input_json data/dataset_coco.json --output_dir data/cocotalk --images_root $IMAGE_ROOT
```


`prepro_feats.py` extract the resnet101 features (both fc feature and last conv feature) of each image. The features are saved in `data/cocotalk_fc` and `data/cocotalk_att`, and resulting files are about 200GB.

(Check the prepro scripts for more options, like other resnet models or other attention sizes.)

**Warning**: the prepro script will fail with the default MSCOCO data because one of their images is corrupted. See [this issue](https://github.com/karpathy/neuraltalk2/issues/4) for the fix, it involves manually replacing one image in the dataset.

### Download Bottom-up features (Skip if you are using resnet features)

Download pre-extracted feature from [link](https://github.com/peteanderson80/bottom-up-attention). You can either download adaptive one or fixed one.

For example:
```
mkdir data/bu_data; cd data/bu_data
wget https://storage.googleapis.com/bottom-up-attention/trainval.zip
unzip trainval.zip

```

Then:

```bash
python script/make_bu_data.py --output_dir data/cocobu
```

This will create `data/cocobu_fc`, `data/cocobu_att` and `data/cocobu_box`. If you want to use bottom-up feature, you can just follow the following steps and replace all cocotalk with cocobu.

## Flickr30k.

It's similar.

```
python scripts/prepro_labels.py --input_json data/dataset_flickr30k.json --output_json data/f30ktalk.json --output_h5 data/f30ktalk

python scripts/prepro_ngrams.py --input_json data/dataset_flickr30k.json --dict_json data/f30ktalk.json --output_pkl data/f30k-train --split train
```

This is to generate the coco-like annotation file for evaluation using coco-caption.

```
python scripts/prepro_reference_json.py --input_json data/dataset_flickr30k.json --output_json data/f30k_captions4eval.json
```

### Feature extraction

For resnet feature, you can do the same thing as COCO.

For bottom-up feature, you can download from [link](https://github.com/kuanghuei/SCAN)

`wget https://scanproject.blob.core.windows.net/scan-data/data.zip`

and then convert to a pth file using the following script:

```
import numpy as np
import os
import torch
from tqdm import tqdm

out = {}
def transform(id_file, feat_file):
  ids = open(id_file, 'r').readlines()
  ids = [_.strip('\n') for _ in ids]
  feats = np.load(feat_file)
  assert feats.shape[0] == len(ids)
  for _id, _feat in tqdm(zip(ids, feats)):
    out[str(_id)] = _feat

transform('dev_ids.txt', 'dev_ims.npy')
transform('train_ids.txt', 'train_ims.npy')
transform('test_ids.txt', 'test_ims.npy')

torch.save(out, 'f30kbu_att.pth')
```