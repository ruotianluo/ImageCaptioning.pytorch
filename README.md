# Neuraltalk2-pytorch

There's something difference compared to neuraltalk2.
- Instead of using random split, we use karparthy's split.
- Instead of including the convnet in the model, we use preprocessed features.
- Use resnet101; the same way as in self-critical (the preprocessing code may have bug, haven't tested yet)

# TODO:
- eval code for arbitrary images
- Other models

# Requirements
Python 2.7 (may work for python 3), pytorch

# Train your own network on COCO
**(Almost identical to neuraltalk2)**

Great, first we need to some preprocessing. Head over to the `coco/` folder and run the IPython notebook to download the dataset and do some very simple preprocessing. The notebook will combine the train/val data together and create a very simple and small json file that contains a large list of image paths, and raw captions for each image, of the form:

```
[{ "file_path": "path/img.jpg", "captions": ["a caption", "a second caption of i"tgit ...] }, ...]
```

Once we have this, we're ready to invoke the `prepro.py` script, which will read all of this in and create a dataset (an hdf5 file and a json file) ready for consumption in the Lua code. For example, for MS COCO we can run the prepro file as follows:

```bash
$ python prepro_split.py --input_json .../dataset_coco.json --output_json data/cocotalk.json --output_h5 data/cocotalk --images_root ...
```

You need to download dataset_coco.json from Karparthy's homepage.

This is telling the script to read in all the data (the images and the captions), allocate 5000 images for val/test splits respectively, and map all words that occur <= 5 times to a special `UNK` token. The resulting `json` and `h5` files are about 30GB and contain everything we want to know about the dataset.

**Warning**: the prepro script will fail with the default MSCOCO data because one of their images is corrupted. See [this issue](https://github.com/karpathy/neuraltalk2/issues/4) for the fix, it involves manually replacing one image in the dataset.

**(Copy end.)**

```bash
$ python train.py --input_json coco/cocotalk.json --input_json data/cocotalk.json --input_fc_h5 data/cocotalk_fc.h5 --save_checkpoint_every 2000 --val_images_use 3200
```

The train script will take over, and start dumping checkpoints into the folder specified by `checkpoint_path` (default = current folder). For more options, see `opts.py`.

If you'd like to evaluate BLEU/METEOR/CIDEr scores during training in addition to validation cross entropy loss, use `--language_eval 1` option, but don't forget to download the [coco-caption code](https://github.com/tylin/coco-caption) into `coco-caption` directory.

**A few notes on training.** To give you an idea, with the default settings one epoch of MS COCO images is about 7500 iterations. 1 epoch of training (with no finetuning - notice this is the default) takes about 45 minutes and results in validation loss ~2.7 and CIDEr score of ~0.5. By iteration 50,000 CIDEr climbs up to about 0.65 (validation loss at about 2.4). 

### Caption images after training

Now place all your images of interest into a folder, e.g. `blah`, and run
the eval script:

```bash
$ python eval.py --model model.ckpt-**** --infos_path infos_<id>.pkl --image_folder <image_folder> --num_images 10
```

This tells the `eval` script to run up to 10 images from the given folder. If you have a big GPU you can speed up the evaluation by increasing `batch_size` (default = 1). Use `-num_images -1` to process all images. The eval script will create an `vis.json` file inside the `vis` folder, which can then be visualized with the provided HTML interface:

```bash
$ cd vis
$ python -m SimpleHTTPServer
```

Now visit `localhost:8000` in your browser and you should see your predicted captions.

**Beam Search**. Beam search is enabled by default because it increases the performance of the search for argmax decoding sequence. However, this is a little more expensive, so if you'd like to evaluate images faster, but at a cost of performance, use `--beam_size 1`. ~~For example, in one of my experiments beam size 2 gives CIDEr 0.922, and beam size 1 gives CIDEr 0.886.~~

**Running on MSCOCO images**. If you train on MSCOCO (see how below), you will have generated preprocessed MSCOCO images, which you can use directly in the eval script. In this case simply leave out the `image_folder` option and the eval script and instead pass in the `input_h5`, `input_json` to your preprocessed files.

