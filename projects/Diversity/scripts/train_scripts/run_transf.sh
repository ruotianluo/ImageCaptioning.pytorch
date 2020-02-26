#! /bin/sh

# fix warmup

id="transf"
ckpt_path="log_"$id
if [ ! -d $ckpt_path ]; then
  mkdir $ckpt_path
fi
if [ ! -f $ckpt_path"/infos_"$id".pkl" ]; then
start_from=""
else
start_from="--start_from "$ckpt_path
fi

python train.py --id $id --caption_model transformer --noamopt --noamopt_warmup 10000 --label_smoothing 0.0 --input_json data/cocotalk.json --input_label_h5 data/cocotalk_label.h5 --input_fc_dir data/cocobu_fc --input_att_dir data/cocobu_att --seq_per_img 5 --batch_size 20 --beam_size 1 --learning_rate 5e-4 --num_layers 6 --input_encoding_size 512 --rnn_size 2048 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --checkpoint_path $ckpt_path $start_from --save_checkpoint_every 3000 --language_eval 1 --val_images_use 5000 --max_epochs 15 

#if [ ! -d xe/$ckpt_path ]; then
#cp -r $ckpt_path xe/
#fi

#python train.py --id $id --caption_model transformer --reduce_on_plateau --input_json data/cocotalk.json --input_label_h5 data/cocotalk_label.h5 --input_fc_dir data/cocobu_fc --input_att_dir data/cocobu_att.lmdb --input_box_dir data/cocobu_box --seq_per_img 5 --batch_size 10 --beam_size 1 --learning_rate 1e-5 --num_layers 6 --input_encoding_size 512 --rnn_size 2048 --checkpoint_path $ckpt_path $start_from --save_checkpoint_every 3000 --language_eval 1 --val_images_use 5000 --self_critical_after 10
