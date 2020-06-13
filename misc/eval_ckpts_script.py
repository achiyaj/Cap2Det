import subprocess
import argparse
import os


def main(model_dir, config_filename, start_epoch, end_epoch, steps_interval, num_eval_examples, epochs_to_eval_list):
    if epochs_to_eval_list is None:
        epochs_to_eval = range(start_epoch, end_epoch + 1, steps_interval)
        log_file = f'eval_log/{model_dir}_{start_epoch}_to_{end_epoch}_{num_eval_examples}_examples.eval_script.log'
    else:
        epochs_to_eval = [int(x) for x in epochs_to_eval_list[0].split(',')]
        log_file = f'eval_log/{model_dir}_top_{len(epochs_to_eval)}_epochs_{num_eval_examples}_examples.eval_script.log'

    # initialize file
    with open(log_file, 'w'):
        pass

    with open(log_file, 'a') as out_f:
        for num_steps in epochs_to_eval:
            ckpt_filename = os.path.join('logs', model_dir, 'ckpts', f'model.ckpt-{num_steps}.meta')
            if os.path.isfile(ckpt_filename):
                out_f.write(f'Starting to eval for step {num_steps}!\n')
                args = ['python', 'train/predict.py', '--alsologtostderr', '--evaluator=coco', '--run_once',
                        f'--pipeline_proto=configs/{config_filename}.pbtxt',
                        f'--model_dir=logs/{model_dir}', f'--max_eval_examples={num_eval_examples}',
                        '--label_file=data/coco_label.txt', f'--eval_log_dir=logs/{model_dir}/eval_det',
                        '--vocabulary_file=data/coco_open_vocab.txt', f'--ckpt_num={num_steps}'
                        ]
                subprocess.call(' '.join(args), stdout=out_f, stderr=out_f, shell=True)
                out_f.write(f'Finished eval for step {num_steps}!\n')
            else:
                out_f.write(f'No ckpt file was find for epoch {num_steps}\n')

            out_f.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--config_filename', type=str, required=True)
    parser.add_argument('--start_epoch', type=int, default=None)
    parser.add_argument('--end_epoch', type=int, default=None)
    parser.add_argument('--steps_interval', type=int, default=2000)
    parser.add_argument('--num_eval_examples', type=int, default=1000)
    parser.add_argument('--epochs_to_eval_list', nargs='*', help='epochs_to_eval', default=None)
    args = parser.parse_args()
    main(args.model_dir, args.config_filename, args.start_epoch, args.end_epoch, args.steps_interval,
         args.num_eval_examples, args.epochs_to_eval_list)
