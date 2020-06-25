import json
import random

sgs_path = 'data/sg_extraction/vacancy_train.json'
rels_vocab_path = 'data/sg_extraction/sg_rels.json'
coco_extend_match_path = 'data/coco_label_synonyms.txt'
num_imgs_to_visl = 50
html_head_str = '<html>\n<head><title>{Rels Visualization}</title></head>\n\t<body>\n{}\t</body>\n</html>'
html_body_str = '\t<hr>\n\t<img src = {}"\t<br>{}\n/>'

def main():
    sgs_data = json.load(open(sgs_path))
    rels_vocab = list(json.load(open(rels_vocab_path)).keys())
    coco_synonyms_lines = open(coco_extend_match_path).readlines()
    synonyms_dict = {}
    for cur_line in coco_synonyms_lines:
        cur_label, synonyms_line = cur_line.strip().split('\t')
        synonyms = synonyms_line.split(',')
        for synonym in synonyms:
            synonyms_dict[synonym] = cur_label

    sgs_ids = list(sgs_data.keys())
    random.shuffle(sgs_ids)
    num_visl_imgs = 0
    visl_data = []
    for sg_id in sgs_ids:
        cur_img_data = ""
        cur_sg = sgs_data[sg_id]
        abc = 123
        for rel_objs, rel_label in cur_sg['relations']:
            rel_obj1_idx, rel_obj2_idx = rel_objs[1:-1].split(', ')
            obj1_label = cur_sg['objects'][rel_obj1_idx]['label']
            obj2_label = cur_sg['objects'][rel_obj2_idx]['label']
            if rel_label in rels_vocab and obj1_label in synonyms_dict and obj2_label in synonyms_dict:
                cur_img_data += f"{obj1_label} {rel_label} {obj2_label}, "

        if len(cur_img_data) > 0:
            visl_data.append((sg_id, cur_img_data))
            num_visl_imgs += 1
            if num_visl_imgs >= num_imgs_to_visl:
                break

    # build HTML file
    img_html_codes = []
    for img_data in visl_data:
        img_html_codes.append()


if __name__ == '__main__':
    main()
