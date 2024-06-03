import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,roc_auc_score,precision_score,recall_score
import torch
from scipy.special import softmax
import numpy as np
is_cuda = torch.cuda.is_available()


import string
import jieba


root_path = '/home/gene/Documents/Senti/AI-Detector'
dataset_path = os.path.join(root_path, 'ExampleData/Comment')

# load training data
# data format:
# {
#   "comment": "这个牌子的糯玉米?，买了几次了，个头大份量足，味道好有口感，一次蒸一根就可以了，搞活动优惠价买的，挺划算的，东北发货，几天才收到", 
#   "product": "玉米", 
#   "label": "human"
# }
human_data_list = json.load(open(os.path.join(dataset_path, 'human_comment_list.json'), 'r'))
gpt_data_list = json.load(open(os.path.join(dataset_path, 'GPT_comment_list.json'), 'r'))
training_data_list = human_data_list + gpt_data_list

# word entropy
word_prob = json.load(open(os.path.join(dataset_path, 'word_entropy.json')))

# training data format: [{'comment': 'xxx'}, {'comment': 'xxx'}, ...]
RawTexts = [{'comment': data_item['comment']} for data_item in training_data_list]

# gpt2 for prelexity features
from transformers import AutoTokenizer, AutoModelForCausalLM
gpt2_path = '/data/pretrained/gpt2-chinese'
gpt2_tokenizer = AutoTokenizer.from_pretrained(gpt2_path)
gpt2_model = AutoModelForCausalLM.from_pretrained(gpt2_path) # CausalLM最后有线性层，hidden_size->vocab_size的映射，gpt2-large为(1280, 50257), gpt2-chinese为（768, 21128)
gpt2_model = gpt2_model.cuda()

# 调用模型计算perplexity，使用gpt2的loss，即负对数似然，越小越好
def sent_scoring(model, tokenizer, text):
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
    input_ids = input_ids.to('cuda')
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    sentence_prob = loss.item()
    return sentence_prob

GPTScores = []
for i in range(len(RawTexts)):
    score = {}
    data = RawTexts[i]
    for key in data:
        score[key] = sent_scoring(gpt2_model, gpt2_tokenizer,data[key])
    GPTScores.append(score)




