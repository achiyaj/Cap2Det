import matplotlib.pyplot as plt

log_file_path = 'log/coco17_sg_extend_match_obj_sg_loss_w_3e-3_att_sg_loss_w_1e-3_retrain.eval.log'
graph_path = f'map_plots/{log_file_path[4:]}.png'
num_oicr_iters = 3


def main():
    log_file_lines = open(log_file_path).readlines()
    eval_end_lines = [i for i, x in enumerate(log_file_lines) if x.startswith('INFO:tensorflow:Summary is written.')]
    eval_start_lines = [i for i, x in enumerate(log_file_lines) if
                        x.startswith('INFO:tensorflow:Start to evaluate checkpoint')][:len(eval_end_lines)]
    eval_ckpts_nums = [int(x[x.rfind('model.ckpt-') + 11:-2]) for x in log_file_lines if
                       x.startswith('INFO:tensorflow:Start to evaluate checkpoint')][:len(eval_end_lines)]

    map_scores = {num_oicr_iter: [] for num_oicr_iter in range(num_oicr_iters + 1)}
    for start_line_idx, end_line_idx in zip(eval_start_lines, eval_end_lines):
        relevant_eval_lines = log_file_lines[start_line_idx: end_line_idx]
        oicr_iter_start_lines = [i for i, x in enumerate(relevant_eval_lines) if x.endswith('predict.py:567] \n')]
        for num_oicr_iter, cur_start_line in enumerate(oicr_iter_start_lines):
            cur_score = float(relevant_eval_lines[cur_start_line + 2].split(' ')[3][:-2]) * 100
            map_scores[num_oicr_iter].append(cur_score)

    plt.plot(eval_ckpts_nums, map_scores[3])
    plt.grid()
    axes = plt.gca()
    axes.set_ylim([0, 10])
    plt.savefig(graph_path)


if __name__ == '__main__':
    main()
