import json
from tqdm import tqdm
from collections import Counter

# COCO Captions
sgs_file = 'data/sg_extraction/vacancy_train.json'
output_file = 'data/sg_extraction/coco_{}_stats.json'
obj_att_pairs_output_file = 'data/sg_extraction/coco_obj_att_pairs_stats.json'
num_occurrences_file = 'data/sg_extraction/coco_extend_match_occurences.json'

# Conceptual Captions
# sgs_file = '/specific/netapp5_2/gamir/datasets/ConceptualCaptions/scene_graphs/vacancy/train_data.json'
# output_file = 'data/sg_extraction/cc_{}_stats.json'

# Flickr30
# sgs_file = 'raw-data-flickr30k/flickr30_sgs.json'
# output_file = 'data/sg_extraction/flickr30_{}_stats.json'
# obj_att_pairs_output_file = 'data/sg_extraction/flickr30_{}_obj_att_pairs_stats.json'

att_categories_file = 'data/sg_extraction/att_categories.json'
objects_file = 'data/coco_label_synonyms.txt'
relations_file = 'data/sg_extraction/relations_dict.json'


def main():
    att_categories = json.load(open(att_categories_file))
    att2cat = {att: cat for cat, cat_atts in att_categories.items() for att in cat_atts.keys()}
    sgs_dict = json.load(open(sgs_file))
    att_stats = {cat_name: {key: 0 for key in cat_atts.keys()} for cat_name, cat_atts in att_categories.items()}
    relevant_rels = list(json.load(open(relations_file)).keys())
    rel_stats = {rel: 0 for rel in relevant_rels}

    relevant_objs = []
    all_obj_labels = []
    synonym_to_obj = {}
    with open(objects_file, "r") as fid:
        lines = fid.readlines()
        for line in lines:
            cur_synonyms = line[line.find('\t'):].strip().split(',')
            relevant_objs += cur_synonyms
            obj_label = line[:line.find('\t')]
            all_obj_labels.append(obj_label)
            for obj_synonym in cur_synonyms:
                synonym_to_obj[obj_synonym] = obj_label

    obj_att_pairs_stats = {obj_label: [] for obj_label in all_obj_labels}
    extend_match_counts = {obj_label: 0 for obj_label in all_obj_labels}

    for sg in tqdm(list(sgs_dict.values())):
        # att stats
        cur_atts = []
        for obj_data in sg['objects'].values():
            obj_label_words = obj_data['label'].split(' ')
            if not any([x in relevant_objs for x in obj_label_words]):
                continue
            cur_atts += obj_data['attributes']

        for att in cur_atts:
            if att not in att2cat:
                continue
            cur_category = att2cat[att]
            if att in att_stats[cur_category]:
                att_stats[cur_category][att] += 1
            else:
                att_stats[cur_category][att] = 1

        # rel stats
        for rel_objs, rel_label in sg['relations'].items():
            obj1, obj2 = rel_objs[1:-1].split(', ')
            if sg['objects'][obj1]['label'] in relevant_objs and \
                    sg['objects'][obj2]['label'] in relevant_objs and \
                    rel_label in relevant_rels:
                rel_stats[rel_label] += 1

        # obj-att pairs stats
        for obj_data in sg['objects'].values():
            obj_label = obj_data['label']
            if obj_label not in relevant_objs:
                continue
            for att_label in obj_data['attributes']:
                if att_label in att2cat:
                    obj_att_pairs_stats[synonym_to_obj[obj_label]].append(att_label)

    # extend_match occurences count
    print('Calculating num exatnd_match occurrences!')
    merged_sg_sents_dict = {}
    for sg_id, sg in sgs_dict.items():
        cur_img_id = sg_id[:sg_id.find('_')]
        if cur_img_id in merged_sg_sents_dict:
            merged_sg_sents_dict[cur_img_id] += sg['sent']
        else:
            merged_sg_sents_dict[cur_img_id] = sg['sent']

    for captions_str in tqdm(list(merged_sg_sents_dict.values())):
        cur_sent_objects = set()
        for synonym_str, obj_label in synonym_to_obj.items():
            if synonym_str in captions_str:
                cur_sent_objects.add(obj_label)

        for obj_label in cur_sent_objects:
            extend_match_counts[obj_label] += 1

    rel_stats = {k: v for k, v in sorted(rel_stats.items(), key=lambda item: item[1], reverse=True)}
    extend_match_counts = {k: v for k, v in sorted(extend_match_counts.items(), key=lambda item: item[1], reverse=True)}
    for obj_name, obj_atts in obj_att_pairs_stats.items():
        obj_atts_counter = dict(Counter(obj_atts))
        obj_atts_counter = {k: v for k, v in sorted(obj_atts_counter.items(), key=lambda item: item[1], reverse=True)}
        obj_att_pairs_stats[obj_name] = obj_atts_counter

    for att_cat, cat_freqs in att_stats.items():
        att_stats[att_cat] = {k: v for k, v in sorted(cat_freqs.items(), key=lambda item: item[1], reverse=True)}

    with open(output_file.format('att'), 'w') as out_f:
        json.dump(att_stats, out_f, indent=4)

    with open(output_file.format('rel'), 'w') as out_f:
        json.dump(rel_stats, out_f, indent=4)

    with open(obj_att_pairs_output_file, 'w') as out_f:
        json.dump(obj_att_pairs_stats, out_f, indent=4)

    with open(num_occurrences_file, 'w') as out_f:
        json.dump(extend_match_counts, out_f, indent=4)


if __name__ == '__main__':
    main()
