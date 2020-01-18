import argparse
import h5py
import os
import numpy as np
import json
from tqdm import tqdm


def main(params):

    imgs = json.load(open(params['input_json'], 'r'))
    imgs = imgs['images']
    N = len(imgs)

    if params['fc_input_dir'] is not None:
        print('processing fc')
        with h5py.File(params['fc_output']) as file_fc:
            for i, img in enumerate(tqdm(imgs)):
                npy_fc_path = os.path.join(
                    params['fc_input_dir'],
                    str(img['cocoid']) + '.npy')

                d_set_fc = file_fc.create_dataset(
                    str(img['cocoid']), data=np.load(npy_fc_path))
            file_fc.close()

    if params['att_input_dir'] is not None:
        print('processing att')
        with h5py.File(params['att_output']) as file_att:
            for i, img in enumerate(tqdm(imgs)):
                npy_att_path = os.path.join(
                    params['att_input_dir'],
                    str(img['cocoid']) + '.npz')

                d_set_att = file_att.create_dataset(
                    str(img['cocoid']),
                    data=np.load(npy_att_path)['feat'])
            file_att.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_json', required=True, help='input json file to process into hdf5')
    parser.add_argument('--fc_output', default='data', help='output h5 filename for fc')
    parser.add_argument('--att_output', default='data', help='output h5 file for att')
    parser.add_argument('--fc_input_dir', default=None, help='input directory for numpy fc files')
    parser.add_argument('--att_input_dir', default=None, help='input directory for numpy att files')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))

    main(params)