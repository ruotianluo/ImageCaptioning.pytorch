#! /bin/sh
set -m
id=$1



python eval.py --image_root /share/data/vision-greg/coco/ --batch_size 100 --dump_images 0 --num_images -1 --split test  --model log_$id/model-best.pth --only_lang_eval 1 --language_eval 1 --beam_size 5 --sample_n $4 --temperature $2 --sample_method greedy --sample_n_method top$3 --infos_path log_$id/infos_$id-best.pkl --id $5$id"_tp_"$2_$3_$4
