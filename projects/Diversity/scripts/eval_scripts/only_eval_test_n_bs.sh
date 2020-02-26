#! /bin/sh
set -m
id=$1

python eval.py --batch_size 10 --image_root /share/data/vision-greg/coco/ --dump_images 0 --num_images -1 --split test  --model log_$id/model-best.pth --language_eval 1 --only_lang_eval 1 --beam_size 5 --sample_n $3 --temperature $2 --sample_method greedy --sample_n_method bs --infos_path log_$id/infos_$id-best.pkl --id $id"_bs_"$2_$3

