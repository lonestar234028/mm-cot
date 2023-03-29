# +
import json
alls = []
pids = []
prompts = []
with open("./data/scienceqa/problems.json") as f:
    alls = json.load(f)
with open("./data/scienceqa/pid_splits.json") as f:
    pids = json.load(f)

with open("/home/taoli1/mm-cot/models/rationale/predictions_ans_test.json") as f:
    prompts = json.load(f)
# -


print(len(alls))
print(len(pids['test']))
print(len(prompts['preds']))

tests = []
idx2tests = {}
idximg2tests = {}
idx = 0
for k,v in alls.items():
    if k in pids['test']:
        tests.append(v)
        idx2tests[idx] = v
        if v['image'] is not None:
            has_img[k] = v
            idximg2tests[idx] = v
        idx += 1


print(len(tests))
print(("tests[0]),", tests[0]))
# print((prompts['preds'][0]))

has_img = {}
for k,v in alls.items():
    if k in pids['test']:
        if v['image'] is not None:
            has_img[k] = v

print(len(has_img))

for k,v in has_img.items():
    print(k)


i = 99

# +
# print(tests[i]['lecture'] + tests[i]['solution'])


# +
# print(prompts['preds'][i])
# -

from rouge import Rouge
def score_rouge(str1, str2):
    rouge = Rouge(metrics=["rouge-l"])
    scores = rouge.get_scores(str1, str2, avg=True)
    rouge_l = scores['rouge-l']['f']
    return rouge_l


# +
rouges = []
for i in range(len(tests)):
    try:
        s1 = (tests[i]['lecture'] + tests[i]['solution']).replace('\n','n')
        s2 = (prompts['preds'][i])[len('Solution:'):].strip()
        rouge = score_rouge(s1, s2)
        rouges.append(rouge)
       
    except:
#         print(i)
#         print("len(s1):",len(s1))
#         print("len(s2):",len(s2))
#         print("S1:",s1)
#         print("S2:",s2)
        continue
    
#     break
# -


sum(rouges) / len(rouges)

tests[i]['lecture'] + tests[i]['solution']


i

z = 0
for k,v in alls.items():
    if k in pids['test']:
        if len(v['lecture']) == 0 and len(v['solution']) == 0:
            z += 1
print(z)

cp = {}
with open("sqa-cp.json") as f:
    cp = json.load(f)
print(len(cp))

print(cp['180'])

import json
res = {}
with open("models/answer/res_fid_20230327.json") as f:
    res = json.load(f)
res_v1 = {}
with open("models/MM-CoT-UnifiedQA-base-Answer/predictions_ans_test.json") as f:
    res_v1 = json.load(f)
preds = res['preds']
preds1 = res_v1['preds']
labels = res['labels']

print(len(preds))
print(len(labels))

for i,x in enumerate(preds):
    y = labels[i]
    if x != y and i in idximg2tests and preds1[i] == y:
        print(idx2tests[i])
        print(idximg2tests[i])
        print(x)
        print(y)
        break

for x in labels:
    print(x)
    break


