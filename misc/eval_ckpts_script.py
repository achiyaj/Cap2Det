import subprocess

model_dir = 'coco17_extend_match_retrain_31_05'
config_filename = 'coco17_extend_match.pbtxt'
num_eval_examples = 1
log_file = f'log/{model_dir}.eval_script.log'

with open(log_file, 'a') as out_f:
    for num_steps in range(2000, 500000, 2000):
        print(f'Starting step {num_steps}!')
        args = ['python', 'train/predict.py', '--alsologtostderr', '--evaluator=coco', '--run_once',
                f'--pipeline_proto=configs/{config_filename}',
                f'--model_dir=logs/{model_dir}', f'--max_eval_examples={num_eval_examples}',
                '--label_file=data/coco_label.txt',
                f'--eval_log_dir=logs/{model_dir}/eval_det', '--vocabulary_file=data/coco_open_vocab.txt',
                f'--ckpt_num={num_steps}']
        cur_args = args
        process = subprocess.call(' '.join(cur_args), stdout=out_f, stderr=out_f, shell=True)
        # process.wait()
