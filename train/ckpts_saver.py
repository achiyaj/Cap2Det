import os
from glob import glob
import time
from shutil import copyfile
import numpy as np

def get_ckpt_num_steps(ckpt_filename):
    return int(ckpt_filename[:-6].split('model.ckpt-')[-1])

def get_latest_ckpt(ckpts_dir):
    ckpts = [x.split('/')[-1] for x in glob(os.path.join(ckpts_dir, 'model.ckpt-*.index'))]
    ckpts_nums = [get_ckpt_num_steps(ckpt) for ckpt in ckpts]
    return ckpts[np.argmax(ckpts_nums)]

def ckpts_saver(model_dir):
    os.makedirs(os.path.join(model_dir, 'ckpts'), exist_ok=True)
    while True:
        latest_ckpt_filename = get_latest_ckpt(model_dir)
        if not os.path.isfile(os.path.join(model_dir, 'ckpts', latest_ckpt_filename)):
            copyfile(os.path.join(model_dir, latest_ckpt_filename),
                     os.path.join(model_dir, 'ckpts', latest_ckpt_filename))
            meta_ckpt_filename = latest_ckpt_filename.replace('index', 'meta')
            copyfile(os.path.join(model_dir, meta_ckpt_filename), os.path.join(model_dir, 'ckpts', meta_ckpt_filename))
            data_ckpt_filename = latest_ckpt_filename.replace('index', 'data-00000-of-00001')
            copyfile(os.path.join(model_dir, data_ckpt_filename), os.path.join(model_dir, 'ckpts', data_ckpt_filename))
            copyfile(os.path.join(model_dir, 'checkpoint'), os.path.join(model_dir, 'ckpts', 'checkpoint'))
        time.sleep(100)
