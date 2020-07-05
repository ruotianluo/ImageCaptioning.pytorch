# Advanced

## Ensemble

Current ensemble only supports models which are subclass of AttModel. Here is example of the script to run ensemble models. The `eval_ensemble.py` assumes the model saving under `log_$id`.

```
python eval_ensemble.py --dump_json 0 --ids model1 model2 model3 --weights 0.3 0.3 0.3 --batch_size 1 --dump_images 0 --num_images 5000 --split test --language_eval 1 --beam_size 5 --temperature 1.0 --sample_method greedy --max_length 30
```

## BPE

```
python scripts/prepro_labels.py --input_json data/dataset_coco.json --output_json data/cocotalk_bpe.json --output_h5 data/cocotalk_bpe --symbol_count 6000
```

Doesn't seem to help improve performance.

## Use lmdb instead of a folder of countless files

It's known that some file systems do not like a folder with a lot of single files. However, in this project, the default way of saving precomputed image features is to save each image feature as an individual file.

Usually, for COCO, once all the features have been cached on the memory (basically after the first epoch), then the time for reading data is negligible. However, for much larger dataset like ConceptualCaptioning, since the number of features is too large and the memory cannot fit all image features, this results in extremely slow data loading and is always slow even passing one epoch.

For that dataset, I used lmdb to save all the features. Although it is still slow to load the data, it's much better compared to saving individual files.

To generate lmdb file from a folder of features, check out `scripts/dump_to_lmdb.py` which is borrowed from [Lyken17/Efficient-PyTorch](https://github.com/Lyken17/Efficient-PyTorch/tools).

I believe the current way of using lmdb in `dataloader.py` is far from optimal. I tried methods in tensorpack but failed to make it work. (The idea was to ready by chunk, so that the lmdb loading can load a chunk at a time, reducing the time for ad hoc disk visiting.)

## new self critical

This "new self critical" is borrowed from "Variational inference for monte carlo objectives". The only difference from the original self critical, is the definition of baseline.

In the original self critical, the baseline is the score of greedy decoding output. In new self critical, the baseline is the average score of the other samples (this requires the model to generate multiple samples for each image).

To try self critical on updown model, you can run

`python train.py --cfg configs/updown_nsc.yml`

This yml file can also provides you some hint what to change to use new self critical.

# SCST in Topdown Bottomup paper

In Topdown bottomup paper, instead of random sampling when SCST, they use beam search. To do so, you can try:

`python train.py --id fc_tdsc --cfg configs/fc_rl.yml --train_sample_method greedy --train_beam_size 5 --max_epochs 30 --learning_rate 5e-6`

## Sample n captions

When sampling, set `sample_n` to be greater than 0. 

## Batch normalization

## Box feature

## Training with pytorch lightning
To run it, you need to install pytorch-lightning, as well as detectron2(for its utility functions).

The benefit of pytorch-lightning is I don't need to take care of the distributed data parallel details. (Although I know how to do this, but it seems lightning is really convenient. Nevertheless I hate the idea that LightningModule is a nn.Module.)

Training script (in fact it's almost identical):
```
python tools/train_pl.py --id trans --cfg configs/transformer.yml
```

Test script:
```
EVALUATE=1 python tools/train_pl.py --id trans --cfg configs/transformer.yml
```