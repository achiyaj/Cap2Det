import matplotlib.pyplot as plt
import numpy as np
import os
import json

log_files = [
    'log/coco17_sg_extend_match_obj_sg_loss_w_3e-3_att_sg_loss_w_1e-3_retrain.eval.log',
    'log/coco17_sg_extend_match_obj_sg_loss_w_3e-3_att_sg_loss_w_1e-3_retrain_210000_to_250000.eval_script.log',
    'log/coco17_sg_extend_match_obj_sg_loss_w_3e-3_att_sg_loss_w_1e-3_retrain_252000_to_300000.eval_script.log',
    'log/coco17_sg_extend_match_obj_sg_loss_w_3e-3_att_sg_loss_w_1e-3_retrain_302000_to_350000.eval_script.log',
    'log/coco17_sg_extend_match_obj_sg_loss_w_3e-3_att_sg_loss_w_1e-3_retrain_352000_to_400000.eval_script.log',
    'log/coco17_sg_extend_match_obj_sg_loss_w_3e-3_att_sg_loss_w_1e-3_retrain_402000_to_450000.eval_script.log',
    'log/coco17_sg_extend_match_obj_sg_loss_w_3e-3_att_sg_loss_w_1e-3_retrain_452000_to_500000.eval_script.log',
    'log/coco17_sg_extend_match_obj_sg_loss_w_3e-3_att_sg_loss_w_1e-3_retrain_cont_from_500K_online_eval.log',
    ]
# log_files = [
#     'log/coco17_extend_match_retrain_31_05_2000_to_100000.eval_script.log',
#     'log/coco17_extend_match_retrain_31_05_102000_to_150000.eval_script.log',
#     'log/coco17_extend_match_retrain_31_05_152000_to_200000.eval_script.log',
#     'log/coco17_extend_match_retrain_31_05_202000_to_250000.eval_script.log',
#     'log/coco17_extend_match_retrain_31_05_252000_to_300000.eval_script.log',
#     'log/coco17_extend_match_retrain_31_05_302000_to_350000.eval_script.log',
#     'log/coco17_extend_match_retrain_31_05_352000_to_400000.eval_script.log',
#     'log/coco17_extend_match_retrain_31_05_402000_to_450000.eval_script.log',
#     'log/coco17_extend_match_retrain_31_05_452000_to_500000.eval_script.log',
#     'log/coco17_extend_match_retrain_31_05_cont_from_500K_online_eval.log',
#     ]

# model_name = 'coco17_extend_match_retrain_31_05'
model_name = 'coco17_sg_extend_match_obj_sg_loss_w_3e-3_att_sg_loss_w_1e-3_retrain'
graph_path = f'map_plots/{model_name}/plot.png'
print_path = f'map_plots/{model_name}/output.txt'
summary_path = f'map_plots/{model_name}/FILENAME_summary.json'
num_oicr_iters = 3
num_top_steps_to_print = 5


def parse_eval_file(filename):
    log_file_lines = open(filename).readlines()
    eval_end_lines = [i for i, x in enumerate(log_file_lines) if
                      x.startswith('INFO:tensorflow:Summary is written.') or x.startswith('Starting eval for step ')
                      or x.startswith('Starting to eval for step ')]
    eval_start_lines = [i for i, x in enumerate(log_file_lines) if
                        x.startswith('INFO:tensorflow:Start to evaluate checkpoint')][:len(eval_end_lines)]
    for line_num, eval_end_line in enumerate(eval_end_lines):
        if eval_end_line > eval_start_lines[0]:
            eval_end_lines = eval_end_lines[line_num:]
            break
    eval_ckpts_nums = [int(x[x.rfind('model.ckpt-') + 11:-2]) for x in log_file_lines if
                       x.startswith('INFO:tensorflow:Start to evaluate checkpoint')][:len(eval_end_lines)]

    map_scores = {str(num_oicr_iter): [] for num_oicr_iter in range(num_oicr_iters + 1)}
    for start_line_idx, end_line_idx in zip(eval_start_lines, eval_end_lines):
        relevant_eval_lines = log_file_lines[start_line_idx: end_line_idx]
        oicr_iter_start_lines = [i for i, x in enumerate(relevant_eval_lines) if x.endswith('predict.py:567] \n') or
                                 x.endswith('predict.py:592] \n')]
        for num_oicr_iter, cur_start_line in enumerate(oicr_iter_start_lines):
            cur_score = float(relevant_eval_lines[cur_start_line + 2].split(' ')[3][:-2]) * 100
            map_scores[str(num_oicr_iter)].append(cur_score)

    return map_scores, eval_ckpts_nums


def main():
    # if not os.path.isfile(summary_path):
    #     all_map_scores = {str(num_oicr_iter): [] for num_oicr_iter in range(num_oicr_iters + 1)}
    #     eval_ckpts_nums = []
    #     for log_file in log_files:
    #         cur_map_scores, cur_eval_ckpts_nums = parse_eval_file(log_file)
    #         eval_ckpts_nums += cur_eval_ckpts_nums
    #         for num_oicr_iter, cur_iter_map_scores in cur_map_scores.items():
    #             all_map_scores[num_oicr_iter] += cur_iter_map_scores
    #     json.dump({'map_scores': all_map_scores, 'ckpts_nums': eval_ckpts_nums}, open(summary_path, 'w'), indent=4)
    # else:
    #     model_data = json.load(open(summary_path))
    #     all_map_scores = model_data['map_scores']
    #     eval_ckpts_nums = model_data['ckpts_nums']

    all_map_scores = {str(num_oicr_iter): [] for num_oicr_iter in range(num_oicr_iters + 1)}
    eval_ckpts_nums = []
    for log_file in log_files:
        log_file_filename = log_file.split('/')[-1]
        cur_summary_path = summary_path.replace('FILENAME', log_file_filename)
        if not os.path.isfile(cur_summary_path):
            cur_map_scores, cur_eval_ckpts_nums = parse_eval_file(log_file)
            with open(cur_summary_path, 'w') as out_f:
                json.dump({'map_scores': cur_map_scores, 'ckpts_nums': cur_eval_ckpts_nums}, out_f, indent=4)
        else:
            cur_file_data = json.load(open(cur_summary_path))
            cur_map_scores = cur_file_data['map_scores']
            cur_eval_ckpts_nums = cur_file_data['ckpts_nums']

        eval_ckpts_nums += cur_eval_ckpts_nums
        for num_oicr_iter, cur_iter_map_scores in cur_map_scores.items():
            all_map_scores[num_oicr_iter] += cur_iter_map_scores

    plt.plot(eval_ckpts_nums, all_map_scores['3'])
    plt.grid()
    axes = plt.gca()
    axes.set_ylim([0, 12])
    minor_ticks = np.arange(0, 12, 0.5)
    axes.set_yticks(minor_ticks, minor=True)
    axes.grid(which='minor', alpha=0.2)
    plt.savefig(graph_path)

    with open(print_path, 'w') as out_f:
        for oicr_step, map_scores in all_map_scores.items():
            map_scores_sorted_indices = np.argsort(map_scores)
            top_scores = np.array(map_scores)[map_scores_sorted_indices[::-1]][:num_top_steps_to_print]
            top_steps = np.array(eval_ckpts_nums)[map_scores_sorted_indices[::-1]][
                        :num_top_steps_to_print]

            out_str = f'For OICR step {oicr_step}, the best mAP scores are {top_scores} at steps ' \
                f'{top_steps}, respectively.'
            print(out_str)
            out_f.write(out_str + '\n')


if __name__ == '__main__':
    main()
