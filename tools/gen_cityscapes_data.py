from os import listdir, makedirs
from os.path import isfile, join, exists
from PIL import Image
from zipfile import ZipFile

import glob
import sys


def _gen_dict(root_dir):
    trainset_path = join(root_dir, 'train')
    valset_path = join(root_dir, 'val')
    all_trainval_files = glob.glob(join(trainset_path, '*/*')) + glob.glob(join(valset_path, '*/*'))
    trainval_map = {}
    for e in all_trainval_files:
        key = '_'.join(e.split('/')[-1].split('_')[0:2])
        if key in trainval_map:
            trainval_map[key].append(e)
        else:
            trainval_map[key] = [e]

    for key, item in trainval_map.items():
        trainval_map[key] = sorted(trainval_map[key])

    update_dict = {}
    for key, item in trainval_map.items():
        if len(trainval_map[key]) > 30:
            this_list = trainval_map[key]
            trainval_map[key] = this_list[0:30]
            for i in range(1, len(this_list) // 30):
                new_key = ''.join(key.split('_')) + 'Reorg' + '_' + str('{:06d}'.format(i-1))
                update_dict[new_key] = this_list[i * 30 : (i + 1) * 30]
    trainval_map.update(update_dict)

    testset_path = join(root_dir, 'test')
    all_test_files = glob.glob(join(testset_path, '*/*'))
    test_map = {}
    for e in all_test_files:
        key = '_'.join(e.split('/')[-1].split('_')[0:2])
        if key in test_map:
            test_map[key].append(e)
        else:
            test_map[key] = [e]

    for key, item in test_map.items():
        test_map[key] = sorted(test_map[key])

    update_dict = {}
    for key, item in test_map.items():
        if len(test_map[key]) > 30:
            this_list = test_map[key]
            test_map[key] = this_list[0:30]
            for i in range(1, len(this_list) // 30):
                new_key = ''.join(key.split('_')) + 'Reorg' + '_' + str('{:06d}'.format(i-1))
                update_dict[new_key] = this_list[i * 30 : (i + 1) * 30]
    test_map.update(update_dict)

    return trainval_map, test_map

def _reorganize(target_dir, target_zip_dir, file_map):
    file_list = []
    if not exists(target_dir):
        makedirs(target_dir)
    if not exists(target_zip_dir):
        makedirs(target_zip_dir)

    number_files = len(file_map)
    for idx, (key, files) in zip(range(number_files), file_map.items()):
        with ZipFile(join(target_zip_dir, key+'.zip'), 'w') as zipf:
            assert len(files) == 30
            for n, item in zip(range(30), files):
                try:
                    im = Image.open(item)
                    im = im.resize((512, 256))
                except:
                    print('####################Failed to open/resize {}/{}#################################'.format(key, item))
                else:
                    arcname =  str('{:06d}'.format(n)) + '_' +item.split('/')[-1].split('_')[-1]
                    if not exists(join(target_dir, key)):
                        makedirs(join(target_dir, key))
                    save_path = join(target_dir, key, arcname)
                    im.save(save_path)
                    zipf.write(save_path, arcname=arcname)
        file_list.append(join(key, '.zip'))
        sys.stdout.write("\r{}/{}>".format(idx, number_files))
        sys.stdout.flush()

    return file_list

def main():
    trainval_map, test_map = _gen_dict('/data/yizhou/cityscape/leftImg8bit_sequence')
    trainval_list = _reorganize('/data/yizhou/cityscape/leftImg8bit_sequence_resized/',
                                '/data/yizhou/cityscape/leftImg8bit_sequence_resized_zip/', trainval_map)
    test_list = _reorganize('/data/yizhou/cityscape/leftImg8bit_sequence_resized/',
                                '/data/yizhou/cityscape/leftImg8bit_sequence_resized_zip/', test_map)

    with open('trainval_list.text', 'w') as outfile:
        outfile.write('\n'.join(trainval_list))
    with open('test_list.text', 'w') as outfile:
        outfile.write('\n'.join(test_list))


if __name__ == '__main__':
    main()
