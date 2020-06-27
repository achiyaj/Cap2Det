import json
import random
import os
from core import plotlib
import cv2
from tqdm import tqdm

sgs_path = 'data/sg_extraction/vacancy_val.json'
rels_vocab_path = 'data/sg_extraction/sg_rels.json'
coco_extend_match_path = 'data/coco_label_synonyms.txt'
visl_html_path = 'data/sg_extraction/val_rels_visualization.html'
imgs_path = '/specific/netapp5_2/gamir/achiya/vqa/Cap2Det_1st_attempt/raw-data-coco/val_imgs/{}.jpg'
num_imgs_to_visl = 100
html_head_str = '<html>\n<head><title>Rels Visualization</title></head>\n\t<body>\nHTML_BODY\t</body>\n</html>'
html_body_str = '\t<hr>\n\t<img src="data:image/jpg;base64,{}">\n\t<br>{}</br>\n'


def main():
    sgs_data = json.load(open(sgs_path))
    sgs_aggregated_by_img = {}
    for sg_id, sg in sgs_data.items():
        img_id = sg_id.split('_')[0]
        if img_id in sgs_aggregated_by_img:
            sgs_aggregated_by_img[img_id].append(sg)
        else:
            sgs_aggregated_by_img[img_id] = [sg]
    rels_vocab = list(json.load(open(rels_vocab_path)).keys())
    coco_synonyms_lines = open(coco_extend_match_path).readlines()
    synonyms_dict = {}
    for cur_line in tqdm(coco_synonyms_lines, desc='Building Scene Graphs dictionary!'):
        cur_label, synonyms_line = cur_line.strip().split('\t')
        synonyms = synonyms_line.split(',')
        for synonym in synonyms:
            synonyms_dict[synonym] = cur_label

    img_ids = list(sgs_aggregated_by_img.keys())
    random.shuffle(img_ids)
    num_visl_imgs = 0
    visl_data = []
    for img_id in img_ids:
        cur_img_path = imgs_path.format(img_id)
        if not os.path.isfile(cur_img_path):
            continue
        cur_img_data = []
        for cur_sg in sgs_aggregated_by_img[img_id]:
            for rel_objs, rel_label in cur_sg['relations'].items():
                rel_obj1_idx, rel_obj2_idx = rel_objs[1:-1].split(', ')
                obj1_label = cur_sg['objects'][rel_obj1_idx]['label']
                obj2_label = cur_sg['objects'][rel_obj2_idx]['label']
                if rel_label in rels_vocab and obj1_label in synonyms_dict and obj2_label in synonyms_dict:
                    cur_img_data.append(f"{synonyms_dict[obj1_label]} {rel_label} {synonyms_dict[obj2_label]}")

        if len(cur_img_data) > 0:
            cur_img_data = ", ".join(cur_img_data)
            visl_data.append((img_id, cur_img_data))
            num_visl_imgs += 1
            if num_visl_imgs >= num_imgs_to_visl:
                break

    # build HTML file
    img_html_strs = []
    for img_data in visl_data:
        sg_id, cur_img_data = img_data
        img_id = sg_id.split('_')[0]
        img_path = imgs_path.format(img_id)
        im = cv2.imread(img_path)
        im_base64 = plotlib._py_convert_to_base64(im)
        img_html_strs.append(html_body_str.format(im_base64, cur_img_data))

    html_text = html_head_str.replace('HTML_BODY', ''.join(img_html_strs))
    with open(visl_html_path, 'w') as out_f:
        out_f.write(html_text)


if __name__ == '__main__':
    main()
