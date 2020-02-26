#! /bin/sh

id="fc"
ckpt_path="log_"$id
if [ ! -d $ckpt_path ]; then
  mkdir $ckpt_path
fi
if [ ! -f $ckpt_path"/infos_"$id".pkl" ]; then
start_from=""
else
start_from="--start_from "$ckpt_path
fi

python train.py --id $id --caption_model newfc --input_json data/cocotalk.json --input_label_h5 data/cocotalk_label.h5 --input_fc_dir data/cocobu_fc --input_att_dir data/cocobu_att --seq_per_img 5 --batch_size 50 --beam_size 1 --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --checkpoint_path $ckpt_path $start_from --save_checkpoint_every 3000 --language_eval 1 --val_images_use 5000 --max_epochs 30 

#if [ ! -d xe/$ckpt_path ]; then
#cp -r $ckpt_path xe/
#fi

#python train.py --id $id --caption_model att2in2 --input_json data/cocotalk.json --input_label_h5 data/cocotalk_label.h5 --input_fc_dir data/cocobu_fc --input_att_dir data/cocobu_att --input_box_dir data/cocobu_box --seq_per_img 5 --batch_size 10 --beam_size 1 --learning_rate 5e-5 --checkpoint_path $ckpt_path $start_from --save_checkpoint_every 500 --language_eval 1 --val_images_use 5000 --self_critical_after 29 
