# Analysis of diversity-accuracy tradeoff in image captioning [[arxiv]](https://arxiv.org/abs/2002.11848)

## Abstract

We investigate the effect of different model architectures, training objectives, hyperparameter settings and decoding procedures on the diversity of automatically generated image captions. Our results show that 1) simple decoding by naive sampling, coupled with low temperature is a competitive and fast method to produce diverse and accurate caption sets; 2) training with CIDEr-based reward using Reinforcement learning harms the diversity properties of the resulting generator, which cannot be mitigated by manipulating decoding parameters. In addition, we propose a new metric AllSPICE for evaluating both accuracy and diversity of a set of captions by a single value.

## AllSPICE

The instruction of AllSPICE has been added to [ruotianluo/coco-caption](https://github.com/ruotianluo/coco-caption). Read the paper for more details of this metric.

## Reproduce the figures

In [drive](https://drive.google.com/open?id=1TILv8GXM0dIcjWnrM5V2D7tJvmqupf49), we provide the original evaluation results. 
To get the figures (the exact same scores) in the paper, you can run `plot.ipynb`.

## Training evaluation scripts

### Training
To train the model used in the main paper, run `run_a2i2.sh` first then run `run_a2i2_npg.sh 1` or `run_a2i2_sf_npg.sh 0` (RL) and `run_a2i2_npg.sh 0` (XE). To get XE+RL, run `run_a2i2_npg.sh x` where x is the weighting factor.

Similar for a2i2l, fc, transf if you want to reproduce the results in appendix.

#### Pretrained models
I also provide pretrained models [link](https://drive.google.com/open?id=1HdEzL-3Bl-uwALlwonLBxyd1zuen2DCO). Note that even with the same model, it's not guaranteed to get the same number for diversity scores because there is randomness in sampling. However, from my experience the numbers are usually close.

### Evaluation

`only_gen_test_n_*.sh` generates the caption sets and `only_eval_test_n_*.sh` evaluates the results.

`*` corresponds to different sampling methods. Check each scripts to see what arguments can be specified. Here is an example:

In `only_gen_test_n_dbst.sh a2i2_npg_0 0.3 1 5`, `a2i2_npg_0` is the model id, `0.3` is the diversity lambda, `1` is the sampling temperature, `5` is the sample size.

## Reference
If you find this work helpful, please cite this paper:

```
@article{luo2020analysis,
  title={Analysis of diversity-accuracy tradeoff in image captioning},
  author={Luo, Ruotian and Shakhnarovich, Gregory},
  journal={arXiv preprint arXiv:2002.11848},
  year={2020}
}
```