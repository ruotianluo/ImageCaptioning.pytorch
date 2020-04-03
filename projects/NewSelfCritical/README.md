# A Better Variant of Self-Critical Sequence Training [[arxiv]](http://arxiv.org/abs/2003.09971)

## Abstract

In this work, we present a simple yet better variant of Self-Critical Sequence Training. We make a simple change in the choice of baseline function in REINFORCE algorithm. The new baseline can bring better performance with no extra cost, compared to the greedy decoding baseline.

## Intro

This "new self critical" is borrowed from "Variational inference for monte carlo objectives". The only difference from the original self critical, is the definition of baseline.

In the original self critical, the baseline is the score of greedy decoding output. In new self critical, the baseline is the average score of the other samples (this requires the model to generate multiple samples for each image).

To try "new self critical" on updown model, you can run

`python train.py --cfg configs/updown_nsc.yml`

This yml file can also provides you some hint what to change to use new self critical.

## My 2 cents

From my experience, this new self critical always works better than SCST. So don't hesitate to use it.

Recent paper meshed-memory-transformer also uses such baseline (their formulation is slightly different from mine, but mathematically they are equivalent). The difference is they use beam search during training instead of sampling; this is following Topdown bottomup paper. However, based on my experiments on both their codebase and my codebase, sampling is better than beam search during training.

(And also, by the way, if using beam search, average reward is not a valid anymore because it's dependent on the samples.)

## Reference
If you find this work helpful, please cite this paper:

```
@article{luo2020better,
  title={A Better Variant of Self-Critical Sequence Training},
  author={Luo, Ruotian},
  journal={arXiv preprint arXiv:2003.09971},
  year={2020}
}
```



