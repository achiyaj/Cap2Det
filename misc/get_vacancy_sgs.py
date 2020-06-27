from tqdm import tqdm
from collections import Counter
import json
from pdb import set_trace as trace
from multiprocessing import Pool

# add vacancy SG parser to path
import sys
sys.path.insert(0, '/specific/netapp5_2/gamir/achiya/vqa/misc/SceneGraphParser')
import sng_parser

captions_path = 'raw-data-flickr30k/results_20130124.token'
sgs_output_path = 'raw-data-flickr30k/flickr30_sgs.json'
NUM_PROCESSES = 8
CHUNK_SIZE = 1000
EXTRACTION_TYPE = 'vacancy'


def get_lines_from_file(input_file):
    with open(input_file) as f:
        lines = f.readlines()
    # return [x.split('\t')[-1].strip() for x in lines]
    return [x.strip() for x in lines]


def get_pretty_graph(parser, sent):
    pretty_graph = {'sent': sent, 'objects': {}, 'relations': {}}
    graph = parser.parse(sent)
    for obj_id, obj in enumerate(graph['entities']):
        cur_label = obj['lemma_head']
        cur_atts = [x['lemma_span'] for x in obj['modifiers'] if x['dep'] == 'amod']
        pretty_graph['objects'][obj_id] = {'label': cur_label, 'attributes': cur_atts}

    for rel in graph['relations']:
        obj, subj = rel['object'], rel['subject']
        pretty_graph['relations'][str((subj, obj))] = rel['lemma_relation']

    return pretty_graph


def get_graphs(input):
    start_idx, sgs_chunk = input
    parser = sng_parser.Parser('spacy', model='en')
    # graphs = {start_idx + i: get_pretty_graph(parser, sent) for i, sent in enumerate(sgs_chunk)}
    graphs = {}
    for cur_line in sgs_chunk:
        img_name, sent = cur_line.split('\t')
        sg_id = img_name[:-6] + '_' + img_name[-1]
        graph = get_pretty_graph(parser, sent)
        graphs[sg_id] = graph
    return graphs


def sort_dict(in_dict):
    return {k: v for k, v in sorted(in_dict.items(), key=lambda item: item[1], reverse=True)}


def main():
    if EXTRACTION_TYPE == 'vacancy':
        lines = get_lines_from_file(captions_path)

        p = Pool(NUM_PROCESSES)
        chunks = [(i * CHUNK_SIZE, lines[CHUNK_SIZE * i: CHUNK_SIZE * (i + 1)]) for i in
                  range(int(len(lines) / CHUNK_SIZE) + 1)]
        outputs = list(tqdm(p.imap(get_graphs, chunks), total=len(chunks)))
        all_graphs = {key: value for cur_dict in outputs for key, value in cur_dict.items()}

        with open(sgs_output_path, 'w') as sgs_f:
            json.dump(all_graphs, sgs_f, indent=2)


if __name__ == '__main__':
    main()
