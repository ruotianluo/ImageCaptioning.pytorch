# Advanced

## Ensemble

Current ensemble only supports models which are subclass of AttModel. Here is example of the script to run ensemble models. The `eval_ensemble.py` assumes the model saving under `log_$id`.

```
python eval_ensemble.py --dump_json 0 --ids model1,model2,model3 --weights 0.3,0.3,0.3 --batch_size 1 --dump_images 0 --num_images 5000 --split test --language_eval 1 --beam_size 5 --temperature 1.0 --sample_method greedy --max_length 30
```

## Batch normalization

## Box feature