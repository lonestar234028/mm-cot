# +
import json
alls = []
pids = []
prompts = []
with open("./data/scienceqa/problems.json") as f:
    alls = json.load(f)
with open("./data/scienceqa/pid_splits.json") as f:
    pids = json.load(f)
import torch
import requests
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

from lavis.common.gradcam import getAttMap
from lavis.models import load_model_and_preprocess



print("len(alls):",len(alls))
print("len(pids['test'],",len(pids['test']))
has_img = {}

tests = {}
for k,v in alls.items():
    if k in pids['test']:
        tests[int(k)] = v
        if v['image'] is not None:
            has_img[k] = v
print("len(tests),",len(tests))
print("len(has_img),",len(has_img))
# print((prompts['preds'][0]))
qid2captions = {}

magic = 85
from tqdm import tqdm

# setup device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




model, vis_processors, txt_processors = load_model_and_preprocess(name="pnp_vqa", model_type="base", is_eval=True, device=device)


id2caption = {}

for i, v in tqdm(has_img.items()):
    # if i < magic:
    #     continue
    # if i > magic:
    #     break
    try:
        img_uri = "./test/" + str(i) + '/image.png'
        raw_image = Image.open(img_uri).convert('RGB')   
        question = v['question']
        print("question,",question)

        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        question = txt_processors["eval"](question)
        samples = {"image": image, "text_input": [question]}

        samples = model.forward_itm(samples=samples)

        samples = model.forward_cap(samples=samples, num_captions=50, num_patches=20)
        captions = samples['captions'][0]
        maxlen = 10 if len(captions) > 10 else len(captions)
        captions = captions[:maxlen]
        id2caption[i] = captions
    except Exception as err:
        print("exception occured:",i,v)
        print("Error: {}:{}:{}".format(i,err,v))
        continue

with open('sqa-cp.json','w') as f:
    json.dump(id2caption, f)
