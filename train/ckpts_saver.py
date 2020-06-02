import os
from glob import glob
import time
from shutil import copyfile


def get_ckpt_filenames(ckpts_dir):
    return [x.split('/')[-1] for x in glob(os.path.join(ckpts_dir, 'model.ckpt-*.index'))]


def ckpts_saver(model_dir):
    os.makedirs(os.path.join(model_dir, 'ckpts'), exist_ok=True)
    while True:
        ckpt_filenames = get_ckpt_filenames(model_dir)
        for ckpt_filename in ckpt_filenames:
            if not os.path.isfile(os.path.join(model_dir, 'ckpts', ckpt_filename)):
                copyfile(os.path.join(model_dir, ckpt_filename),
                         os.path.join(model_dir, 'ckpts', ckpt_filename))
                meta_ckpt_filename = ckpt_filename.replace('index', 'meta')
                copyfile(os.path.join(model_dir, meta_ckpt_filename), os.path.join(model_dir, 'ckpts', meta_ckpt_filename))
                data_ckpt_filename = ckpt_filename.replace('index', 'data-00000-of-00001')
                copyfile(os.path.join(model_dir, data_ckpt_filename), os.path.join(model_dir, 'ckpts', data_ckpt_filename))
                copyfile(os.path.join(model_dir, 'checkpoint'), os.path.join(model_dir, 'ckpts', 'checkpoint'))
        time.sleep(100)
