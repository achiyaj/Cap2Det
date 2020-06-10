import os
from glob import glob
import time
from shutil import copyfile
import argparse
import datetime

MAX_TRIES_TIME = 10000
SLEEP_TIME = 100
LOG_FILE = ''


def get_ckpt_filenames(ckpts_dir):
    return [x.split('/')[-1] for x in glob(os.path.join(ckpts_dir, 'model.ckpt-*.meta'))]


def get_train_step(ckpt_filename):
    return int(ckpt_filename.split('/')[-1].split('.')[1][5:])


def main(model_dir):
    with open(os.path.join(model_dir, 'ckpts_saver_out.txt'), 'w') as log_file:
        try:
            os.makedirs(os.path.join(model_dir, 'ckpts'), exist_ok=True)
            time_from_last_save = 0
            while True:
                ckpt_filenames = get_ckpt_filenames(model_dir)
                found_new_ckpts = False
                for ckpt_filename in ckpt_filenames:
                    if not os.path.isfile(os.path.join(model_dir, 'ckpts', ckpt_filename)):
                        ckpt_step = get_train_step(ckpt_filename)
                        copyfile(os.path.join(model_dir, ckpt_filename),
                                 os.path.join(model_dir, 'ckpts', ckpt_filename))
                        index_ckpt_filename = ckpt_filename.replace('meta', 'index')
                        copyfile(os.path.join(model_dir, index_ckpt_filename), os.path.join(model_dir, 'ckpts', index_ckpt_filename))
                        data_ckpt_filename = ckpt_filename.replace('meta', 'data-00000-of-00001')
                        copyfile(os.path.join(model_dir, data_ckpt_filename), os.path.join(model_dir, 'ckpts', data_ckpt_filename))
                        copyfile(os.path.join(model_dir, 'checkpoint'),
                                 os.path.join(model_dir, 'ckpts', f'checkpoint-{ckpt_step}'))
                        log_file.write(f'{datetime.datetime.now()}: Copied ckpt file {ckpt_filename} to ckpts dir\n')
                        time_from_last_save = 0
                        found_new_ckpts = True

                if not found_new_ckpts:
                    # no new ckpt
                    log_file.write(f'{datetime.datetime.now()}: No new ckpts\n')
                    time_from_last_save += SLEEP_TIME
                    if time_from_last_save > MAX_TRIES_TIME:
                        return

                log_file.flush()
                time.sleep(SLEEP_TIME)

        except Exception as e:
            log_file.write(f'{datetime.datetime.now()}: Exception! {e}\n')
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    args = parser.parse_args()
    main(model_dir=args.model_dir)
