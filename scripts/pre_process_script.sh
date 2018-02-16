mkdir data
cd data
wget https://download.pytorch.org/models/resnet101-5d3b4d8f.pth -O resnet101.pth
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
unzip train2014.zip
unzip val2014.zip
wget http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
unzip caption_datasets.zip
python ./prepro_labels.py --input_json ./dataset_coco.json --output_json ./cocotalk.json --output_h5 ./cocotalk
python ./prepro_feats.py --input_json ./dataset_coco.json --output_dir ./cocotalk --images_root ./ --model_root ./
# in case of corrunption due to bad file closure:
# h5clear --status data/cocotalk_att/feats_att.h5 
# h5clear --status data/coco_preprocessed/data/cocotalk_att/feats_att.h5

