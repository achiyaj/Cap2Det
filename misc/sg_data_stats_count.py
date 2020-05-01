import json
from tqdm import tqdm

coco_train_sgs_file = 'data/sg_extraction/vacancy_train.json'
att_categories_file = 'data/sg_extraction/att_categories.json'
output_file = 'data/sg_extraction/coco_{}_stats.json'
objects_file = 'data/coco_label_synonyms.txt'
relations_file = 'data/sg_extraction/relations_dict.json'


def main():
    att_categories = json.load(open(att_categories_file))
    att2cat = {att: cat for cat, cat_atts in att_categories.items() for att in cat_atts.keys()}
    coco_train_sgs = json.load(open(coco_train_sgs_file))
    att_stats = {cat_name: {key: 0 for key in cat_atts.keys()} for cat_name, cat_atts in att_categories.items()}
    relevant_rels = list(json.load(open(relations_file)).keys())
    rel_stats = {rel: 0 for rel in relevant_rels}

    relevant_objs = []
    with open(objects_file, "r") as fid:
        lines = fid.readlines()
        for line in lines:
            relevant_objs += line[line.find('\t'):].strip().split(',')

    for sg in tqdm(coco_train_sgs.values()):
        # att stats
        cur_atts = [att for obj_data in sg['objects'].values() for att in obj_data['attributes'] if
                    obj_data['label'] in relevant_objs]
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

    rel_stats = {k: v for k, v in sorted(rel_stats.items(), key=lambda item: item[1], reverse=True)}
    for att_cat, cat_freqs in att_stats.items():
        att_stats[att_cat] = {k: v for k, v in sorted(cat_freqs.items(), key=lambda item: item[1], reverse=True)}

    with open(output_file.format('att'), 'w') as out_f:
        json.dump(att_stats, out_f, indent=4)

    with open(output_file.format('rel'), 'w') as out_f:
        json.dump(rel_stats, out_f, indent=4)


if __name__ == '__main__':
    main()
