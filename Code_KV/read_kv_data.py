"""
    读取kv-data
"""
import csv
import argparse
import sys
import random
import numpy as np
from word_process_method import *
from collections import defaultdict
from tqdm import tqdm
def get_maxlen(*paths):
    maxlen =defaultdict(int)
    for path in paths:
        with open(path,'r') as input_files:
            fields=['question','qn_entities','ans_entities','sources','relations','targets']
            reader =csv.DictReader(input_files,delimiter='\t',fieldnames=fields)
            for row in tqdm(reader):
                example = {}
                example['question'] = row['question'].split(" ")
                example['qn_entities'] =row['qn_entities'].split("|")
                example['ans_entities']=row['ans_entities'].split("|")
                example['sources']=row['sources'].split("|")
                example['relations'] = row['relations'].split("|")
                example['targets'] = row['targets'].split("|")

                maxlen['question'] = max(len(example['question']),maxlen['question'])
                maxlen['qn_entities'] = max(len(example['qn_entities']), maxlen['qn_entities'])
                maxlen['ans_entities'] = max(len(example['ans_entities']), maxlen['ans_entities'])
                maxlen['sources'] = max(len(example['sources']), maxlen['sources'])
                maxlen['relations'] = maxlen['sources']
                maxlen['targets'] = maxlen['sources']
    return maxlen
class DataReader(object):
    def __init__(self,args,maxlen,share_idx=True,data_name=None):
        print("Reading...."+data_name)
        self.share_idx=share_idx
        word_idx = read_file_as_dict(args.word_idx)
        self.word_idx_size = len(word_idx)
        entity_idx = read_file_as_dict(args.entity_idx)
        self.entity_idx_size = len(entity_idx)
        relation_idx = read_file_as_dict(args.relation_idx)
        self.relation_idx_size=len(relation_idx)
        idx=read_file_as_dict(args.idx)
        self.idx = len(idx)
        encoder = idx
        fields = ['question','qn_entities','ans_entities','sources','relations','targets']
        with open(args.input_examples,'r') as input_examples_file:
            reader = csv.DictReader(input_examples_file,delimiter='\t',fieldnames=fields)
            self.maxlen = maxlen
            self.num_examples = 0
            examples = []
            for row in tqdm(reader):
                example ={}
                example['question'] = row['question'].split(" ")
                example['qn_entities'] =row['qn_entities'].split("|")
                example['ans_entities'] = row['ans_entities'].split("|")
                example['sources'] = row['sources'].split("|")
                example['relations'] = row['relations'].split("|")
                example['targets']  = row['targets'].split("|")
                self.num_examples +=1
                examples.append(example)
            vec_examples = []
            for it in tqdm(examples):
                vec_example = {}
                for key in it.keys():
                    """if key == 'question':
                        encoder=word_idx
                    elif key == 'relations':
                        encoder=relation_idx
                    else :
                        encoder = entity_idx
                    """
                    if self.share_idx:
                        encoder=idx

                    if key == 'ans_entities':
                        encoder =entity_idx #答案永远是用实体编码!!!

                    #开始编码
                    vec_example[key]=[encoder[w]-1 for w in it[key]]


                    #if key == 'ans_entities':
                        # 答案应该是 0-实体数目-1
                    #    vec_example[key]= [label -1 for label in vec_example[key]]

                vec_examples.append(vec_example)
        print("Finish reading..."+data_name)
        self.vec_examples = vec_examples
    def get_examples(self):
        """
        直接返回编码过后的数据集
        :return:
        """
        return self.vec_examples
    def get_max_lengths(self):
        return self.maxlen
    def get_word_idx_size(self):
        return self.word_idx_size
    def get_relation_idx_size(self):
        return self.relation_idx_size
    def get_entities_size(self):
        return self.entity_idx_size
    def get_idx_size(self):
        return self.idx


