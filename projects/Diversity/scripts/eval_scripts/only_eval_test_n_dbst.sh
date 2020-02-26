#! /bin/sh
set -m
id=$1

python eval.py --batch_size 1 --diversity_lambda $2 --image_root /share/data/vision-greg/coco/ --dump_images 0 --num_images -1 --split test  --model log_$id/model-best.pth --only_lang_eval 1 --language_eval 1 --beam_size 5 --sample_n $4 --temperature $3 --sample_method greedy --sample_n_method dbs --infos_path log_$id/infos_$id-best.pkl --id $id"_dbst_"$2_$3_$4 
